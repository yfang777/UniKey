#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
import yaml

from policy.utils.runtime import cache_environment, egl_device
from policy.policy_training.config import PolicyRuntime, SUPPORTED_POLICIES, read_policy_runtime

from policy.data_generation.parallel import (
    concatenate_datasets,
    scene_paths,
    run_parallel_generation,
)


@dataclass(frozen=True)
class PendingChunk:
    dataset_dir: str
    frames: int


@dataclass(frozen=True)
class GenerationState:
    next_scene_idx: int
    pending_chunks: tuple[PendingChunk, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense Sample Configuration")
    parser.add_argument("--policy", choices=SUPPORTED_POLICIES, required=True)
    parser.add_argument("--device", type=str, required=True, help="Device to use")
    parser.add_argument("--policy_preset", default="default", help="Named preset in the policy YAML")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--representation",
        type=str,
        choices=("rgb",),
        default="rgb",
        help="Must be rgb",
    )
    parser.add_argument(
        "--start_training_steps",
        type=int,
        default=40000,
        help="Training steps for the first fresh-run dataset/checkpoint step",
    )
    parser.add_argument(
        "--incremental_training_steps",
        type=int,
        default=0,
        help="Training steps for each resumed or later dataset/checkpoint step",
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--no_ee", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--add_action_noise", action="store_true")
    parser.add_argument(
        "--incremental_dataset_step_number",
        type=int,
        default=None,
        help="Frame-step increment between cumulative dataset targets",
    )
    dataset_step_group = parser.add_mutually_exclusive_group(required=True)
    dataset_step_group.add_argument("--episodes", type=int,
                                    help="Collect exactly this many successful complete episodes")
    dataset_step_group.add_argument(
        "--start_dataset_step_number",
        type=int,
        default=None,
        help="First dataset/checkpoint frame-step target for a fresh run",
    )
    dataset_step_group.add_argument(
        "--continue_dataset_step_number",
        type=int,
        default=None,
        help="First dataset/checkpoint frame-step target to process when resuming",
    )
    parser.add_argument(
        "--total_dataset_step_number",
        type=int,
        default=None,
        help="Final cumulative frame-step target, inclusive",
    )
    parser.add_argument(
        "--evaluate_dataset_step_number",
        type=int,
        default=None,
        help="Evaluate only when the dataset-step target is a multiple of this value",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=50,
        help="Number of XMLs to roll out per eval policy batch",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--only_generate_training",
        action="store_true",
        help="Run datagen + training only, skip rollout",
    )
    mode_group.add_argument(
        "--only_evaluate",
        action="store_true",
        help="Run rollout only using existing checkpoints",
    )
    mode_group.add_argument(
        "--generate_training_and_evaluate",
        action="store_true",
        help="Run datagen + training + rollout",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Disable rollout regardless of selected mode",
    )
    parser.add_argument("--fsdp_devices", "--fsdp-devices", type=int, default=int(os.environ.get("OPENPI_FSDP_DEVICES", "2")))
    parser.add_argument("--datagen_workers", "--datagen-workers", type=int, default=None)
    parser.add_argument("--datagen_tmp_root", "--datagen-tmp-root", default=os.environ.get("DATAGEN_TMP_ROOT"))
    parser.add_argument("--save_interval", "--save-interval", type=int, default=None)
    parser.add_argument("--scene_dir", type=Path, help="Task scene directory containing train/ and test_uniform/")
    parser.add_argument("--test_scene_count", type=int, default=100)
    parser.add_argument("--training_workers", type=int, default=4)
    return parser.parse_args()


def execution_modes(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    if args.only_generate_training:
        do_generate, do_train, do_eval = True, True, False
    elif args.only_evaluate:
        do_generate, do_train, do_eval = False, False, True
    else:
        do_generate, do_train, do_eval = True, True, True

    if args.skip_evaluation:
        if args.only_evaluate:
            raise ValueError("--skip_evaluation is incompatible with --only_evaluate.")
        do_eval = False

    return do_generate, do_train, do_eval


def dataset_frame_episode_counts(dataset_dir: str | Path) -> tuple[int, int] | None:
    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        total_frames = int(info["total_frames"])
        total_episodes = int(info["total_episodes"])
        if total_frames > 0 and total_episodes > 0:
            return total_frames, total_episodes

    parquets = sorted(dataset_dir.glob("data/chunk-*/*.parquet"))
    if not parquets:
        return None
    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in parquets), len(parquets)


def total_dataset_frames(dataset_dir: str | Path) -> int | None:
    counts = dataset_frame_episode_counts(dataset_dir)
    if counts is None:
        return None
    return counts[0]


def dataset_step_name(step_number: int) -> str:
    return f"dataset_step_{step_number}"


def dataset_step_lower_bound(
    total_frames: int,
    total_episodes: int,
    target_step: int,
) -> float:
    return target_step - 1.1 * (total_frames / total_episodes)


def verify_dataset_step(dataset_dir: str | Path, target_step: int) -> None:
    counts = dataset_frame_episode_counts(dataset_dir)
    if counts is None:
        raise ValueError(
            f"{dataset_step_name(target_step)} dataset missing or empty at {dataset_dir}"
        )

    total_frames, total_episodes = counts
    avg_frames = total_frames / total_episodes
    lower_bound = dataset_step_lower_bound(total_frames, total_episodes, target_step)
    if total_frames < lower_bound:
        raise ValueError(
            f"{dataset_step_name(target_step)} has {total_frames} frames, "
            f"expected > {lower_bound:.0f} "
            f"(target={target_step}, avg_frames_per_episode={avg_frames:.1f})"
        )


def normalize_pending_chunk(item: dict, state_path: Path) -> tuple[PendingChunk, bool]:
    dataset_dir = item.get("dataset_dir", item.get("dir"))
    frames = item.get("frames", item.get("length"))
    if dataset_dir is None or frames is None:
        raise ValueError(
            f"Invalid pending chunk entry in {state_path}: expected "
            "'dataset_dir'/'frames' or legacy 'dir'/'length', got {item}"
        )

    changed = "dataset_dir" not in item or "frames" not in item
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        parts = dataset_path.parts
        if state_path.parent.name in parts:
            prefix_idx = parts.index(state_path.parent.name)
            candidate = state_path.parent.joinpath(*parts[prefix_idx + 1 :])
            if candidate.exists():
                dataset_dir = str(candidate)
                changed = True

    return PendingChunk(dataset_dir=str(dataset_dir), frames=int(frames)), changed


def load_generation_state(state_path: Path, discard_missing_chunks: bool = False) -> GenerationState:
    if not state_path.exists():
        return GenerationState(next_scene_idx=0, pending_chunks=())

    with open(state_path, "r") as f:
        data = json.load(f)

    normalized_chunks = [
        normalize_pending_chunk(item, state_path)
        for item in data["pending_chunks"]
    ]
    pending_chunks = tuple(chunk for chunk, _ in normalized_chunks)
    if discard_missing_chunks:
        available_chunks = tuple(chunk for chunk in pending_chunks if Path(chunk.dataset_dir).exists())
        missing_count = len(pending_chunks) - len(available_chunks)
        if missing_count:
            print(f"Dropping {missing_count} pending chunks removed from temporary storage.")
        pending_chunks = available_chunks
    state = GenerationState(
        next_scene_idx=int(data["next_scene_idx"]),
        pending_chunks=pending_chunks,
    )
    if len(pending_chunks) != len(normalized_chunks) or any(changed for _, changed in normalized_chunks):
        save_generation_state(state_path, state)
    return state


def save_generation_state(state_path: Path, state: GenerationState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".json.tmp")
    data = {
        "next_scene_idx": state.next_scene_idx,
        "pending_chunks": [
            {"dataset_dir": chunk.dataset_dir, "frames": chunk.frames}
            for chunk in state.pending_chunks
        ],
    }
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, state_path)


def build_datagen_args(task: str, task_config: dict) -> argparse.Namespace:
    args = argparse.Namespace()
    for key, value in task_config["tasks"][task].items():
        setattr(args, key, value)
    args.task = task
    return args


def generated_chunks_to_pending(chunk_dirs: list[str]) -> tuple[PendingChunk, ...]:
    chunks = []
    for chunk_dir in chunk_dirs:
        frames = total_dataset_frames(chunk_dir)
        if frames is None:
            print(f"[skip empty chunk] {chunk_dir}")
            continue
        chunks.append(PendingChunk(dataset_dir=chunk_dir, frames=frames))
    return tuple(chunks)


def generate_one_batch(
    state: GenerationState,
    datagen_args: argparse.Namespace,
    scene_prefix: str,
    pool_root: Path,
    state_path: Path,
) -> GenerationState:
    batch_idx = state.next_scene_idx // datagen_args.workers
    start = state.next_scene_idx
    end = start + datagen_args.workers
    scenes = scene_paths(datagen_args.task, scene_prefix, start, end)

    batch_args = copy.copy(datagen_args)
    batch_args.tmp_root = str(pool_root / f"batch_{batch_idx:05d}")
    chunk_dirs = run_parallel_generation(scenes, f"Pool batch {batch_idx}", batch_args, workers=datagen_args.workers)

    next_state = GenerationState(
        next_scene_idx=end,
        pending_chunks=state.pending_chunks + generated_chunks_to_pending(chunk_dirs),
    )
    save_generation_state(state_path, next_state)
    return next_state


def selected_prefix_count(
    pending_chunks: tuple[PendingChunk, ...],
    budget_frames: int,
) -> tuple[int, int]:
    selected_count = 0
    used_frames = 0

    for chunk in pending_chunks:
        if used_frames + chunk.frames > budget_frames:
            break
        selected_count += 1
        used_frames += chunk.frames

    return selected_count, used_frames


def delete_consumed_batch_dirs(
    selected_chunks: tuple[PendingChunk, ...],
    remaining_chunks: tuple[PendingChunk, ...],
) -> None:
    consumed_batch_dirs = {
        str(Path(chunk.dataset_dir).parent)
        for chunk in selected_chunks
    }
    remaining_batch_dirs = {
        str(Path(chunk.dataset_dir).parent)
        for chunk in remaining_chunks
    }

    for batch_dir in consumed_batch_dirs - remaining_batch_dirs:
        shutil.rmtree(batch_dir)
        print(f"Deleted consumed batch dir: {batch_dir}")


def compose_dataset_step(
    state: GenerationState,
    datagen_args: argparse.Namespace,
    scene_prefix: str,
    pool_root: Path,
    state_path: Path,
    prev_dataset_dir: str | None,
    dataset_step_dir: str,
    target_frames: int,
    prev_dataset_frames: int,
    camera_keys: list[str],
) -> GenerationState:
    budget_frames = target_frames - prev_dataset_frames
    prev_dataset_episodes = 0
    if prev_dataset_dir is not None:
        _, prev_dataset_episodes = dataset_frame_episode_counts(prev_dataset_dir)

    if budget_frames <= 0:
        print(
            f"Dataset-step target ({target_frames} frames) already met by previous "
            f"dataset ({prev_dataset_frames} frames); re-concatenating previous only."
        )
        concatenate_datasets([prev_dataset_dir], dataset_step_dir, camera_keys)
        return state

    current_state = state

    while True:
        selected_count, _ = selected_prefix_count(
            current_state.pending_chunks,
            budget_frames,
        )
        if selected_count < len(current_state.pending_chunks):
            break

        current_state = generate_one_batch(
            current_state,
            datagen_args,
            scene_prefix,
            pool_root,
            state_path,
        )

    selected_count, _ = selected_prefix_count(
        current_state.pending_chunks,
        budget_frames,
    )
    if selected_count < len(current_state.pending_chunks):
        used_frames = sum(
            chunk.frames for chunk in current_state.pending_chunks[:selected_count]
        )
        total_frames = prev_dataset_frames + used_frames
        total_episodes = prev_dataset_episodes + selected_count
        if total_episodes == 0 or total_frames < dataset_step_lower_bound(
            total_frames,
            total_episodes,
            target_frames,
        ):
            selected_count += 1

    selected_chunks = current_state.pending_chunks[:selected_count]
    remaining_chunks = current_state.pending_chunks[selected_count:]
    selected_dirs = [chunk.dataset_dir for chunk in selected_chunks]
    source_dirs = ([prev_dataset_dir] if prev_dataset_dir else []) + selected_dirs
    if not source_dirs:
        raise ValueError(
            f"No generated episode fits {dataset_step_name(target_frames)} "
            f"with budget {budget_frames} frames."
        )

    concatenate_datasets(source_dirs, dataset_step_dir, camera_keys)

    next_state = GenerationState(
        next_scene_idx=current_state.next_scene_idx,
        pending_chunks=remaining_chunks,
    )
    save_generation_state(state_path, next_state)
    delete_consumed_batch_dirs(selected_chunks, remaining_chunks)

    selected_frames = [chunk.frames for chunk in selected_chunks]
    if selected_frames:
        print(
            f"Dataset step added {len(selected_frames)} episodes; "
            f"avg episode length = {sum(selected_frames) / len(selected_frames):.1f} "
            f"frames; total new frames = {sum(selected_frames)}"
        )
    else:
        print("Dataset step added 0 episodes.")

    return next_state


def target_step_numbers(
    first_step_number: int,
    total_dataset_step_number: int,
    incremental_dataset_step_number: int,
) -> list[int]:
    if incremental_dataset_step_number <= 0:
        raise ValueError("--incremental_dataset_step_number must be > 0")
    if total_dataset_step_number <= 0:
        raise ValueError("--total_dataset_step_number must be > 0")
    if first_step_number <= 0:
        raise ValueError(
            "--start_dataset_step_number or --continue_dataset_step_number must be > 0"
        )
    if first_step_number > total_dataset_step_number:
        raise ValueError(
            f"First dataset step ({first_step_number}) must be <= "
            f"--total_dataset_step_number ({total_dataset_step_number})"
        )
    if total_dataset_step_number % incremental_dataset_step_number != 0:
        raise ValueError(
            "--total_dataset_step_number must be a multiple of "
            "--incremental_dataset_step_number"
        )
    if first_step_number % incremental_dataset_step_number != 0:
        raise ValueError(
            "First dataset step must be a multiple of "
            "--incremental_dataset_step_number"
        )

    return list(
        range(
            first_step_number,
            total_dataset_step_number + 1,
            incremental_dataset_step_number,
        )
    )


def representation_dir(
    representation: str,
    no_ee: bool,
    add_action_noise: bool,
    use_relative_action: bool,
) -> str:
    rep_dir = f"{representation}_no_ee" if no_ee else representation
    if not use_relative_action:
        rep_dir = f"{rep_dir}_abs_action"
    if add_action_noise:
        rep_dir = f"{rep_dir}_add_action_noise"
    return rep_dir


def write_training_config(
    save_dir: str,
    task: str,
    rep_dir: str,
    step_name: str,
    policy: str,
    training_steps: int,
    policy_runtime: PolicyRuntime,
    representation: str,
    training_workers: int | None = None,
) -> Path:
    config_path = (
        Path(save_dir)
        / task
        / "dense_sample"
        / rep_dir
        / step_name
        / f"config_{policy}_training_steps_{training_steps}.yaml"
    )
    training_config = copy.deepcopy(policy_runtime.config)
    training_config[policy_runtime.policy_key]["setting"]["repo_id"] = task
    training_config[policy_runtime.policy_key]["train"][
        "num_steps_finetune"
    ] = training_steps
    training_config[policy_runtime.policy_key]["train"][
        f"num_steps_{representation}"
    ] = training_steps
    if training_workers is not None:
        training_config[policy_runtime.policy_key]["train"]["num_workers_rgb"] = training_workers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(training_config, f)
    return config_path


def train_command(
    representation: str,
    policy: str,
    dataset_dir: str,
    policy_config_path: str | Path,
    ckpt_dir: str,
    device: str,
    seed: int,
    task: str,
    no_ee_flag: list[str],
    abs_action_flag: list[str],
    train_augmentation_flags: list[str],
    pretrained_ckpt_path: str | None,
    fsdp_devices: int,
    save_interval: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        f"policy.policy_training.{representation}.train.{policy}",
        "--dataset_dir",
        dataset_dir,
        "--policy_config_path",
        str(policy_config_path),
        "--ckpt_save_dir",
        ckpt_dir,
        "--device",
        device,
        "--seed",
        str(seed),
        "--task",
        task,
    ]
    if policy == "Pi05":
        cmd.extend(["--fsdp-devices", str(fsdp_devices)])
        if save_interval is None:
            cmd.append("--no-save_intermediate_ckpts")
        else:
            cmd.extend(["--save_interval", str(save_interval)])
    if pretrained_ckpt_path is not None:
        cmd.extend(["--pretrained_ckpt_path", pretrained_ckpt_path])
    return cmd + no_ee_flag + abs_action_flag + train_augmentation_flags


def cleanup_after_subprocess() -> None:
    gc.collect()


def rollout_command(
    representation: str,
    policy: str,
    module_name: str,
    scene_test_paths: list[str],
    output_dir: Path,
    ckpt_final: str,
    policy_config_path: str | Path,
    device: str,
    task: str,
    no_ee_flag: list[str],
    abs_action_flag: list[str],
    rollout_batch_size: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        f"policy.policy_evaluation.{module_name}",
        "--xml_list",
        *scene_test_paths,
        "--output_dir",
        str(output_dir),
        "--pretrained_ckpt_path",
        ckpt_final,
        "--policy_config_path",
        str(policy_config_path),
        "--device",
        device,
        "--task",
        task,
        "--log_video",
    ]
    cmd.extend(
        [
            "--rollout_batch_size",
            str(rollout_batch_size),
            "--multiprocess_envs",
        ]
    )
    return cmd + no_ee_flag + abs_action_flag


def verify_episode_dataset(dataset_dir: Path, episodes: int) -> tuple[int, int]:
    counts = dataset_frame_episode_counts(dataset_dir)
    if counts is None or counts[1] != episodes:
        raise ValueError(f"Expected exactly {episodes} episodes at {dataset_dir}; found {counts}")
    records = json.loads((dataset_dir / "generation.info").read_text())
    if len(records) != episodes or not all(record["success"] for record in records):
        raise ValueError(f"Dataset contains missing or unsuccessful demonstrations: {dataset_dir}")
    return counts


def collect_episode_dataset(task, task_config, scene_dir, dataset_prefix, episodes, workers, egl_device):
    """Collect successful whole episodes and record their actual frame count."""

    dataset_prefix.mkdir(parents=True, exist_ok=True)
    manifest = dataset_prefix / f"episodes_{episodes}.json"
    if manifest.exists():
        result = json.loads(manifest.read_text())
        verify_episode_dataset(Path(result["dataset_dir"]), episodes)
        return result

    state_path = dataset_prefix / f"episodes_{episodes}_state.json"
    state = load_generation_state(state_path)
    pool_root = dataset_prefix / f"episodes_{episodes}_pool"
    while len(state.pending_chunks) < episodes:
        count = min(workers, episodes - len(state.pending_chunks))
        start = state.next_scene_idx
        scenes = scene_paths(task, scene_dir / "train", start, start + count)
        generation_args = build_datagen_args(task, task_config)
        generation_args.mujoco_egl_device_id = egl_device
        generation_args.tmp_root = str(pool_root / f"batch_{start:06d}")
        chunks = run_parallel_generation(scenes, f"episodes {start}", generation_args, workers=count)
        successful = []
        for name in chunks:
            directory = Path(name)
            records = json.loads((directory / "generation.info").read_text())
            if len(records) != 1 or not records[0]["success"]:
                continue
            frames, _ = verify_episode_dataset(directory, 1)
            successful.append(PendingChunk(str(directory), frames))
        state = GenerationState(start + count, state.pending_chunks + tuple(successful))
        save_generation_state(state_path, state)
        print(f"Collected {len(state.pending_chunks)}/{episodes} successful episodes", flush=True)

    frames = sum(chunk.frames for chunk in state.pending_chunks)
    dataset_dir = dataset_prefix / dataset_step_name(frames)
    camera_keys = [f"observation.images.{name}" for name in task_config["tasks"][task]["camera_names"]]
    if not dataset_dir.exists():
        concatenate_datasets([chunk.dataset_dir for chunk in state.pending_chunks], str(dataset_dir), camera_keys)
    actual_frames, actual_episodes = verify_episode_dataset(dataset_dir, episodes)
    if actual_frames != frames:
        raise ValueError(f"Concatenation changed the frame count: {frames} -> {actual_frames}")
    result = {"dataset_dir": str(dataset_dir.resolve()), "dataset_step": frames,
              "total_frames": actual_frames, "total_episodes": actual_episodes}
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(manifest)
    shutil.rmtree(pool_root)
    return result


def run_episode_training(args):
    if args.episodes <= 0 or args.start_training_steps <= 0 or args.training_workers < 0:
        raise ValueError("Episode/training counts must be positive; training workers must be nonnegative")
    if args.total_dataset_step_number is not None or args.incremental_dataset_step_number is not None:
        raise ValueError("Episode mode does not accept frame-budget arguments")
    workers = args.datagen_workers if args.datagen_workers is not None else 2
    if workers <= 0:
        raise ValueError("--datagen_workers must be positive")
    do_generate, do_train, do_eval = execution_modes(args)
    task_dir = Path(args.save_dir).resolve() / args.task
    scene_dir = args.scene_dir or Path("data/scenes") / args.task
    cache_env = cache_environment(task_dir / "cache")
    for key, value in cache_env.items():
        if key != "WANDB_MODE":
            Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(cache_env)
    egl_id = egl_device(args.device.split()[0])
    env = {**os.environ, "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": egl_id}
    tasks = yaml.safe_load(Path("configurations/tasks.yaml").read_text())
    dataset_prefix = task_dir / "dense_sample/lerobot_dataset"
    if do_generate:
        result = collect_episode_dataset(args.task, tasks, scene_dir, dataset_prefix,
                                         args.episodes, workers, egl_id)
    else:
        result = json.loads((dataset_prefix / f"episodes_{args.episodes}.json").read_text())
        verify_episode_dataset(Path(result["dataset_dir"]), args.episodes)
    runtime = read_policy_runtime(args.policy, args.policy_preset)
    rep_dir = representation_dir(args.representation, args.no_ee, args.add_action_noise,
                                 runtime.use_relative_action)
    run_name = args.policy if args.policy_preset == "default" else f"{args.policy}_{args.policy_preset}"
    step_name = dataset_step_name(result["dataset_step"])
    config_path = write_training_config(args.save_dir, args.task, rep_dir, step_name, run_name,
                                        args.start_training_steps, runtime, args.representation,
                                        training_workers=args.training_workers)
    run_dir = task_dir / "dense_sample" / rep_dir / step_name
    ckpt_dir = run_dir / f"{run_name}_ckpt_seed_{args.seed}"
    checkpoint = ckpt_dir / "final"
    complete = ckpt_dir / "training_complete.json"
    no_ee = ["--no_ee"] if args.no_ee else []
    absolute = [] if runtime.use_relative_action else ["--use_absolute_action"]
    if do_train and not complete.exists():
        subprocess.run(train_command(args.representation, args.policy, result["dataset_dir"], config_path,
                                     str(ckpt_dir), args.device, args.seed, args.task, no_ee, absolute,
                                     ["--add_action_noise"] if args.add_action_noise else [], None,
                                     args.fsdp_devices, args.save_interval), env=env, check=True)
        complete.write_text(json.dumps({"updates": args.start_training_steps}) + "\n")
    if do_eval:
        scenes = scene_paths(args.task, scene_dir / "test_uniform", 0, args.test_scene_count)
        output = run_dir / f"{run_name}_rollout_seed_{args.seed}" / f"{step_name}_test_rollout"
        subprocess.run(rollout_command(args.representation, args.policy, f"{args.policy}_ensemble",
                                      scenes, output, str(checkpoint), config_path, args.device,
                                      args.task, no_ee, absolute, args.rollout_batch_size), env=env, check=True)


def main() -> None:
    args = parse_args()
    if args.episodes is not None:
        run_episode_training(args)
        return
    if args.incremental_dataset_step_number is None or args.total_dataset_step_number is None:
        raise ValueError("Frame mode requires incremental and total dataset frame targets")
    do_generate, do_train, do_eval = execution_modes(args)
    workers = args.datagen_workers if args.datagen_workers is not None else int(
        os.environ.get("DATAGEN_WORKERS", "1" if args.policy == "Pi05" else "10")
    )
    if workers < 1 or args.fsdp_devices < 1:
        raise ValueError("Worker and FSDP device counts must be positive")
    if args.save_interval is not None and args.save_interval <= 0:
        raise ValueError("--save_interval must be positive")
    policy_run_name = args.policy if args.policy_preset == "default" else f"{args.policy}_{args.policy_preset}"
    if args.policy == "Pi05":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.representation != "rgb":
        raise ValueError("Only rgb representation is supported.")
    if args.start_training_steps <= 0:
        raise ValueError("--start_training_steps must be > 0")
    if args.incremental_training_steps < 0:
        raise ValueError("--incremental_training_steps must be >= 0")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout_batch_size must be > 0")
    evaluate_dataset_step_number = (
        args.incremental_dataset_step_number
        if args.evaluate_dataset_step_number is None
        else args.evaluate_dataset_step_number
    )
    if evaluate_dataset_step_number <= 0:
        raise ValueError("--evaluate_dataset_step_number must be > 0")

    with open("configurations/tasks.yaml", "r") as f:
        task_config = yaml.safe_load(f)
    if args.task not in task_config["tasks"]:
        raise ValueError(f"Task '{args.task}' not found in configurations/tasks.yaml")

    policy_runtime = read_policy_runtime(args.policy, args.policy_preset)
    rep_dir = representation_dir(
        args.representation,
        args.no_ee,
        args.add_action_noise,
        policy_runtime.use_relative_action,
    )

    print(
        f"Policy config: chunk_size={policy_runtime.chunk_size}, "
        f"policy_freq={policy_runtime.policy_freq}, "
        f"lerobot_fps={policy_runtime.lerobot_fps}, "
        f"use_relative_action={policy_runtime.use_relative_action}. "
        "Dataset steps are counted as dataset frames. "
        f"Fresh first-round training uses {args.start_training_steps} steps; "
        f"resumed and later rounds use {args.incremental_training_steps} steps. "
        f"Evaluation runs every {evaluate_dataset_step_number} dataset steps."
    )

    start_from_scratch = args.start_dataset_step_number is not None
    first_step_number = (
        args.start_dataset_step_number
        if start_from_scratch
        else args.continue_dataset_step_number
    )
    if (
        args.continue_dataset_step_number is not None
        and args.continue_dataset_step_number <= args.incremental_dataset_step_number
    ):
        raise ValueError(
            "--continue_dataset_step_number must be greater than "
            "--incremental_dataset_step_number; use --start_dataset_step_number "
            "for the first dataset step"
        )

    step_numbers = target_step_numbers(
        first_step_number,
        args.total_dataset_step_number,
        args.incremental_dataset_step_number,
    )
    run_mode = "fresh run" if start_from_scratch else "resume"
    print(
        f"Starting {run_mode} at dataset step {first_step_number}. "
        f"Running dataset-step targets {step_numbers[0]}..{step_numbers[-1]} "
        f"in increments of {args.incremental_dataset_step_number} frames."
    )

    save_dir = Path(args.save_dir)
    cache_root = Path(os.environ.get("CODEGEN_CACHE_DIR", Path.home() / ".cache"))
    cache_env = cache_environment(cache_root)
    for key, path in cache_env.items():
        if key != "WANDB_MODE":
            Path(path).mkdir(parents=True, exist_ok=True)
    os.environ.update(cache_env)

    dataset_prefix = save_dir / args.task / "dense_sample" / "lerobot_dataset"
    state_path = dataset_prefix / "state.json"
    pool_root = (
        Path(args.datagen_tmp_root).expanduser() / args.task / "dense_sample" / "lerobot_dataset" / "pool"
        if args.datagen_tmp_root else dataset_prefix / "pool"
    )
    task_scene_dir = args.scene_dir or Path("data/scenes") / args.task
    scene_prefix = str(task_scene_dir / "train")
    scene_test_prefix = str(task_scene_dir / "test_uniform")
    scene_test_paths = scene_paths(args.task, scene_test_prefix, 0, args.test_scene_count) if do_eval else []
    mujoco_egl_device_id = egl_device(args.device.split()[0])

    datagen_state = None
    datagen_args = None
    camera_keys: list[str] = []
    if do_generate:
        if not start_from_scratch and not state_path.exists():
            raise ValueError(
                f"--continue_dataset_step_number={step_numbers[0]} starts at "
                f"{dataset_step_name(step_numbers[0])} but {state_path} does not exist."
            )
        datagen_state = load_generation_state(state_path, discard_missing_chunks=bool(args.datagen_tmp_root))
        datagen_args = build_datagen_args(args.task, task_config)
        datagen_args.mujoco_egl_device_id = mujoco_egl_device_id
        datagen_args.workers = workers
        camera_keys = [
            f"observation.images.{camera_name}"
            for camera_name in datagen_args.camera_names
        ]

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = mujoco_egl_device_id
    env.update(cache_env)

    no_ee_flag = ["--no_ee"] if args.no_ee else []
    abs_action_representations = {"rgb"}
    abs_action_flag = (
        ["--use_absolute_action"]
        if args.representation in abs_action_representations
        and not policy_runtime.use_relative_action
        else []
    )
    train_augmentation_flags = []
    if args.add_action_noise:
        train_augmentation_flags.append("--add_action_noise")
    module_name = f"{args.policy}_ensemble"

    for target_step in step_numbers:
        current_step_name = dataset_step_name(target_step)
        previous_step = target_step - args.incremental_dataset_step_number
        prev_step = (
            None
            if start_from_scratch and target_step == first_step_number
            else previous_step
        )
        prev_step_name = (
            dataset_step_name(prev_step)
            if prev_step is not None
            else None
        )
        dataset_step_dir = str(dataset_prefix / current_step_name)
        run_dir = save_dir / args.task / "dense_sample" / rep_dir / current_step_name
        training_steps = (
            args.start_training_steps
            if start_from_scratch and target_step == first_step_number
            else args.incremental_training_steps
        )

        for train_seed in [args.seed]:
            ckpt_dir = str(run_dir / f"{policy_run_name}_ckpt_seed_{train_seed}")
            rollout_dir = run_dir / f"{policy_run_name}_rollout_seed_{train_seed}"
            prev_ckpt_dir = (
                save_dir
                / args.task
                / "dense_sample"
                / rep_dir
                / prev_step_name
                / f"{policy_run_name}_ckpt_seed_{train_seed}"
                if prev_step_name is not None
                else None
            )

            print("========================================================")
            print(
                f"STARTING {current_step_name}: target <= {target_step} "
                f"frame steps | SEED {train_seed}"
            )
            print("========================================================")

            if prev_ckpt_dir is not None and do_train:
                prev_ckpt_final = prev_ckpt_dir / "final"
                if not prev_ckpt_final.exists():
                    raise ValueError(
                        f"{prev_step_name} checkpoint not found at {prev_ckpt_final}"
                    )

            if do_generate:
                existing_frames = total_dataset_frames(dataset_step_dir)
                if existing_frames is not None:
                    verify_dataset_step(dataset_step_dir, target_step)
                    print(
                        f"{current_step_name} dataset present with "
                        f"{existing_frames} frames; skipping generation."
                    )
                else:
                    prev_dataset_dir = (
                        str(dataset_prefix / prev_step_name)
                        if prev_step_name is not None
                        else None
                    )
                    if prev_dataset_dir is not None:
                        verify_dataset_step(prev_dataset_dir, prev_step)
                    prev_dataset_frames = (
                        total_dataset_frames(prev_dataset_dir)
                        if prev_dataset_dir is not None
                        else 0
                    )
                    print(
                        f"Composing {current_step_name}: "
                        f"prev_steps={prev_dataset_frames or 0} frames, "
                        f"target={target_step} frames"
                    )
                    datagen_state = compose_dataset_step(
                        state=datagen_state,
                        datagen_args=datagen_args,
                        scene_prefix=scene_prefix,
                        pool_root=pool_root,
                        state_path=state_path,
                        prev_dataset_dir=prev_dataset_dir,
                        dataset_step_dir=dataset_step_dir,
                        target_frames=target_step,
                        prev_dataset_frames=prev_dataset_frames or 0,
                        camera_keys=camera_keys,
                    )
                    actual_frames = total_dataset_frames(dataset_step_dir)
                    print(
                        f"{current_step_name} dataset has {actual_frames} "
                        "frames after composition."
                    )
                    verify_dataset_step(dataset_step_dir, target_step)

            policy_config_path = write_training_config(
                args.save_dir,
                args.task,
                rep_dir,
                current_step_name,
                policy_run_name,
                training_steps,
                policy_runtime,
                args.representation,
                training_workers=args.training_workers,
            )
            print(f"Saved policy config to {policy_config_path}")

            if do_train:
                pretrained_ckpt_path = (
                    str(prev_ckpt_dir / "final")
                    if prev_ckpt_dir is not None
                    else None
                )
                action = "Training from scratch" if pretrained_ckpt_path is None else (
                    f"Finetuning from {prev_step_name}"
                )
                print(f"[{current_step_name} - Seed {train_seed}] {action}...")
                try:
                    subprocess.run(
                        train_command(
                            representation=args.representation,
                            policy=args.policy,
                            dataset_dir=dataset_step_dir,
                            policy_config_path=policy_config_path,
                            ckpt_dir=ckpt_dir,
                            device=args.device,
                            seed=train_seed,
                            task=args.task,
                            no_ee_flag=no_ee_flag,
                            abs_action_flag=abs_action_flag,
                            train_augmentation_flags=train_augmentation_flags,
                            pretrained_ckpt_path=pretrained_ckpt_path,
                            fsdp_devices=args.fsdp_devices,
                            save_interval=args.save_interval,
                        ),
                        env=env,
                        check=True,
                    )
                finally:
                    cleanup_after_subprocess()

            if do_eval and target_step % evaluate_dataset_step_number == 0:
                ckpt_final = f"{ckpt_dir}/final"
                if not Path(ckpt_final).exists():
                    raise ValueError(
                        f"{current_step_name} checkpoint not found at {ckpt_final}; "
                        "cannot evaluate."
                    )
                rollout_output_dir = rollout_dir / f"{current_step_name}_test_rollout"
                rollout_output_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"[{current_step_name} - Seed {train_seed}] "
                    "Running rollout on normal test scenes..."
                )
                try:
                    subprocess.run(
                        rollout_command(
                            representation=args.representation,
                            policy=args.policy,
                            module_name=module_name,
                            scene_test_paths=scene_test_paths,
                            output_dir=rollout_output_dir,
                            ckpt_final=ckpt_final,
                            policy_config_path=policy_config_path,
                            device=args.device,
                            task=args.task,
                            no_ee_flag=no_ee_flag,
                            abs_action_flag=abs_action_flag,
                            rollout_batch_size=args.rollout_batch_size,
                        ),
                        env=env,
                        check=True,
                    )
                finally:
                    cleanup_after_subprocess()
            elif do_eval:
                print(
                    f"[{current_step_name} - Seed {train_seed}] "
                    f"Skipping rollout; {target_step} is not a multiple of "
                    f"{evaluate_dataset_step_number}."
                )


if __name__ == "__main__":
    main()

"""Generate a local LeRobot dataset from successful seeded demonstrations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import pandas as pd
import yaml
from lerobot.datasets.aggregate import aggregate_datasets

from keypoint_program_solver import constraint_helpers
from keypoint_program_solver.solver import Solver
from keypoint_program_solver.utils import CameraRenderers
from policy.scene_generator.generator import make_generator
from policy.scene_generator.task_scene_generators import TASK_SCENE_GENERATORS
from policy.evaluate import bind_helpers, evaluate_episode, load_module
from policy.utils.features import create_dataset, dataset_features


def collect_episode(task, root, seed, episode, stride, max_attempts, mode="stable", correction=None, *, scene_path=None):
    cfg = yaml.safe_load(Path("configurations/tasks.yaml").read_text())
    motion = yaml.safe_load(Path("configurations/keypoint_program_solver/config.yaml").read_text())
    task_cfg = cfg["tasks"][task]
    generator = make_generator(task, cfg) if scene_path is None else None
    features = dataset_features(task, task_cfg["camera_names"], cfg["camera_params"])
    judge = load_module(task_cfg["constraint_judge_path"], "constraints_judge")
    bind_helpers(judge, constraint_helpers)
    solver = Solver(task_cfg, motion)
    attempts = []
    for retry in range(max_attempts):
        index = episode + retry * stride
        scene_seed = seed + index
        scene = scene_path
        if scene is None:
            params = generator.sample_params(1, scene_seed)[0]
            scene = generator.generate_scene(params, root / "scenes", index)
        start = time.monotonic()
        if correction is not None:
            env = correction(solver, scene, judge)
        elif mode == "normal":
            env = solver.execute(scene, seed=scene_seed)
        else:
            env = solver.execute_stable(scene)
        frames = env.lerobot_features
        passed, stage, last_frame = evaluate_episode(
            pd.DataFrame(frames), judge, judge.grasp_stage, task_cfg["rekep_tolerance"]
        )
        elapsed = time.monotonic() - start
        scene_label = str(Path(scene).relative_to(root)) if scene_path is None else str(scene)
        attempt = dict(scene=scene_label, success=passed,
                       last_success_stage=stage, last_success_frame=last_frame, frames=len(frames),
                       simulation_seconds=elapsed, episode_index=episode if passed else None)
        if scene_path is None:
            attempt["seed"] = scene_seed
        attempts.append(attempt)
        print(f"{scene_label}: stage {stage}, {len(frames)} frames, passed={passed}", flush=True)
        if passed:
            break
    else:
        if scene_path is not None:
            return None, attempts
        raise RuntimeError(f"Episode {episode} exhausted its scene attempts: {attempts}")

    if env.cams != task_cfg["camera_names"] or env.robot_variant != task_cfg["robot_variant"]:
        raise ValueError(f"Camera or robot mismatch in {scene}")
    start = time.monotonic()
    shard = root / "shards" / str(episode)
    dataset = create_dataset(shard, task, task_cfg, cfg["camera_params"])
    renderers = CameraRenderers(env.model)
    states = []
    try:
        for frame_index, frame in enumerate(frames):
            # Rebuild the exact kinematic transforms visible at the original sample.
            env.data.qpos[:] = frame["render_qpos"]
            mujoco.mj_kinematics(env.model, env.data)
            mujoco.mj_camlight(env.model, env.data)
            values = {**frame, "success": [1], "elapsed_time": [elapsed]}
            record = {k: np.asarray(values[k], dtype=spec["dtype"]).reshape(spec["shape"])
                      for k, spec in features.items() if spec["dtype"] != "video"}
            for camera in env.cams:
                renderer = renderers.for_camera(camera)
                renderer.update_scene(env.data, camera=camera)
                record[f"observation.images.{camera}"] = renderer.render()
            dataset.add_frame({**record, "task": task_cfg["task_description"]})
            states.append(dict(episode_index=episode, frame_index=frame_index,
                               state_buf=frame["state_buf"], last_gripper_status=frame["last_gripper_status"]))
            if "observation.constraint_context" in frame:
                states[-1]["constraint_context"] = frame["observation.constraint_context"]
        dataset.save_episode(parallel_encoding=False)
    finally:
        renderers.close()
        dataset.finalize()
    pd.DataFrame(states).to_parquet(shard / "states.parquet", index=False)
    attempt["export_seconds"] = time.monotonic() - start
    return shard, attempts


def generate(task, root, episodes, workers, seed, max_attempts, mode="stable", correction=None):
    if mode not in ("stable", "normal"):
        raise ValueError(f"Unknown solver mode {mode!r}")
    root.mkdir(parents=True)
    start = time.monotonic()
    multiprocessing.set_forkserver_preload(["lerobot.datasets.aggregate", "keypoint_program_solver.solver",
                                           "policy.utils.features", "pandas"])
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        jobs = [pool.submit(collect_episode, task, root, seed, i, episodes, max_attempts, mode, correction)
                for i in range(episodes)]
        results = [job.result() for job in jobs]
    shards = [shard for shard, _ in results]
    attempts = [row for _, rows in results for row in rows]
    staging = root / "assembling"
    aggregate_datasets([f"unikey/{task}"] * episodes, f"unikey/{task}", roots=shards,
                       aggr_root=staging, video_files_size_in_mb=1e-6)
    state_dir = staging / "meta/state_buffers/chunk-000"
    state_dir.mkdir(parents=True)
    states = pd.concat([pd.read_parquet(shard / "states.parquet") for shard in shards], ignore_index=True)
    states.to_parquet(state_dir / "file-000.parquet", index=False)
    report = dict(task=task, mode="correction" if correction else mode, workers=workers, seed=seed, episodes=episodes,
                  wall_seconds=time.monotonic() - start, attempts=attempts)
    (staging / "generation.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.rename(root / "dataset")
    shutil.rmtree(root / "shards")
    print(f"{episodes} episodes in {root / 'dataset'}", flush=True)


def main(*, task=None, default_output=None, correction=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if task is None:
        parser.add_argument("--task", choices=TASK_SCENE_GENERATORS, required=True)
    if not correction:
        parser.add_argument("--mode", choices=("stable", "normal"), default="stable",
                            help="Replay recorded targets (stable) or optimize keypoint constraints (normal).")
    parser.add_argument("--output", type=Path, default=default_output, required=default_output is None)
    parser.add_argument("--episodes", type=int, default=1 if task else 10)
    parser.add_argument("--workers", type=int, default=1 if task else 10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=100)
    if task is not None:
        parser.add_argument("--train-steps", type=int, default=0, help="ACT updates after dataset generation")
        if correction is not None:
            parser.add_argument("--base-dataset", type=Path, help="Original demonstrations for corrective co-training")
            parser.add_argument("--checkpoint", type=Path, help="Initial ACT checkpoint for fine-tuning")
    args = parser.parse_args()
    if task is not None and args.train_steps < 0:
        parser.error("--train-steps must be nonnegative")
    if correction is not None and args.train_steps and args.base_dataset is None:
        parser.error("corrective ACT training requires --base-dataset")
    if min(args.episodes, args.workers, args.max_attempts) < 1 or args.seed < 0:
        parser.error("counts must be positive and seed must be nonnegative")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    generate(task or args.task, args.output.resolve(), args.episodes, args.workers, args.seed, args.max_attempts,
             "stable" if correction else args.mode, correction)
    if task is not None and args.train_steps:
        import torch
        from policy.policy_training.train_act import train

        dataset = args.output / "dataset"
        train(args.base_dataset if correction else dataset, args.output / "checkpoints",
              recovery=dataset if correction else None,
              checkpoint=args.checkpoint if correction else None,
              steps=args.train_steps, device="cuda" if torch.cuda.is_available() else "cpu", seed=args.seed)


if __name__ == "__main__":
    main()

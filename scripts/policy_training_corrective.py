"""Collect corrective demonstrations, fine-tune a policy, and evaluate each round.

Uses stable task solutions and absolute actions. New runs live under corrective/.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from policy.data_generation.parallel import scene_paths
from policy.utils.video import SuppressNativeStderr
from policy.utils.runtime import cache_environment, egl_device
from policy.policy_training.config import SUPPORTED_POLICIES, read_policy_runtime


WORKSPACE_TAG = "corrective"
DEFAULT_SUCCESS_RATE_THRESHOLD = 0.99
NO_IMPROVEMENT_ROLLOUT_PATIENCE = 5


def _fmt_float_for_path(x):
    """Encode a float for use in a directory name. Dots -> 'p' so the
    resulting segment is filesystem-friendly. Integers come out without
    a fractional part. Examples:
        0.005 -> '0p005'
        5     -> '5'
        2.0   -> '2p0'
    """
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) >= 1):
        return str(int(x))
    return ("%g" % x).replace(".", "p")


def run_command(cmd_list, cache_env, env=None):
    cmd_str_list = [str(x) for x in cmd_list]
    process_env = os.environ.copy()
    process_env["MUJOCO_GL"] = "egl"
    process_env.update(cache_env)
    if env:
        process_env.update(env)
    subprocess.run(cmd_str_list, check=True, env=process_env)


def dataset_step_name(step_number: int) -> str:
    return f"dataset_step_{step_number}"


def count_dataset_frames(parquet_paths, cutoff=None):
    """Count dataset steps as parquet rows."""
    total = 0
    for pf in parquet_paths:
        ep_len = int(pq.ParquetFile(pf).metadata.num_rows)
        if cutoff is not None:
            ep_len = min(ep_len, cutoff)
        total += ep_len
    return total


def dataset_parquet_paths(dataset_dir):
    return sorted(Path(dataset_dir).glob("data/chunk-*/*.parquet"))


def dataset_frame_count(dataset_dir):
    info_path = Path(dataset_dir) / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        return int(info["total_frames"])
    return count_dataset_frames(dataset_parquet_paths(dataset_dir))


def collect_recovery_data(
    step_name,
    target_recovery_frames,
    devices,
    mujoco_egl_device_ids,
    rollout_dir,
    ckpt_path,
    policy_cfg_path,
    representation,
    policy_name,
    task,
    tasks_config_path,
    pos_diff_threshold,
    policy_tolerance,
    planner_lerobot_fps,
    policy_lerobot_fps,
    policy_infer_freq,
    recovery_seconds,
    recovery_cutoff,
    python_executable,
    seed,
    max_scenes_per_worker,
    log_video,
    rollout_batch_size,
    no_ee=False,
):
    """Spawn workers across GPUs and accumulate recovery parquet paths.

    Ceil-divide the remaining
    target across N GPUs, run all workers in parallel, aggregate their
    results, repeat until accumulated >= target.
    """
    rollout_dir = Path(rollout_dir)
    hunting_dir = rollout_dir / f"{step_name}_hunting_recovery"
    hunting_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = len(devices)
    remaining = target_recovery_frames
    attempt = 0
    worker_id_offset = 0

    collected_parquets = []
    collected_records = []
    aggregate_case_counts = {str(case): 0 for case in range(7)}
    aggregate_policy_pass = 0
    aggregate_policy_evaluated = 0
    aggregate_policy_failed = 0
    aggregate_recovery_attempted = 0
    aggregate_recovery_skipped_after_target = 0
    aggregate_scenes = 0

    while remaining > 0:
        if n_gpus <= 1:
            worker_target = remaining
            spawns = 1
        else:
            worker_target = math.ceil(remaining / n_gpus)
            spawns = n_gpus

        print(f"\n[{step_name}] Recovery collection attempt {attempt} on {spawns} GPUs "
              f"(per-gpu target: {worker_target} frames; remaining: {remaining})")
        worker_rollout_batch_size = max(1, math.ceil(rollout_batch_size / spawns))

        processes = []
        for i in range(spawns):
            worker_id = worker_id_offset + i
            worker_gpu = devices[i % n_gpus]
            worker_log = open(str(hunting_dir / f"worker_{worker_id}.out"), "w")

            cmd = [
                python_executable, "-m", "policy.corrective_data_generation.corrective_worker",
                "--worker_id", str(worker_id),
                "--target_frames", str(worker_target),
                "--ckpt_path", str(ckpt_path),
                "--rollout_dir", str(hunting_dir),
                "--device", worker_gpu,
                "--policy_cfg", str(policy_cfg_path),
                "--representation", representation,
                "--policy_name", policy_name,
                "--task", task,
                "--tasks_config_path", tasks_config_path,
                "--recovery_seconds", str(recovery_seconds),
                "--planner_lerobot_fps", str(planner_lerobot_fps),
                "--policy_lerobot_fps", str(policy_lerobot_fps),
                "--policy_infer_freq", str(policy_infer_freq),
                "--pos_diff_threshold", str(pos_diff_threshold),
                "--policy_tolerance", str(policy_tolerance),
                "--max_scenes", str(max_scenes_per_worker),
                "--seed", str(seed),
                "--rollout_batch_size", str(worker_rollout_batch_size),
            ]
            if log_video:
                cmd.append("--log_video")
            if no_ee:
                cmd.append("--no_ee")

            env = os.environ.copy()
            env["MUJOCO_GL"] = "egl"
            env["MUJOCO_EGL_DEVICE_ID"] = mujoco_egl_device_ids[worker_gpu]
            p = subprocess.Popen(cmd, env=env, stdout=worker_log, stderr=subprocess.STDOUT)
            processes.append((worker_id, p, worker_log))

        for worker_id, p, out_file in processes:
            p.wait()
            out_file.close()
            gc.collect()

        new_parquets_this_attempt = []
        for i in range(spawns):
            worker_id = worker_id_offset + i
            result_path = hunting_dir / f"worker_{worker_id}_results.json"
            if not result_path.exists():
                aggregate_case_counts["3"] += 1
                continue
            with open(result_path, "r") as f:
                res = json.load(f)
            if res["case"] == 3:
                aggregate_case_counts["3"] += 1
                continue
            worker_parquets = res["collected_recovery_parquets"]
            new_parquets_this_attempt.extend(worker_parquets)
            collected_parquets.extend(worker_parquets)
            collected_records.extend(res.get("collected_recovery_records", []))
            for case_str, cnt in res["case_counts"].items():
                aggregate_case_counts[case_str] += cnt
            aggregate_policy_evaluated += res["policy_evaluated_count"]
            aggregate_policy_pass += res["policy_passed_count"]
            aggregate_policy_failed += res["policy_failed_count"]
            aggregate_recovery_attempted += res["recovery_attempted_count"]
            aggregate_recovery_skipped_after_target += (
                res["recovery_skipped_after_target_count"]
            )
            aggregate_scenes += res["scenes_generated"]

        attempt_collected_frames = count_dataset_frames(
            new_parquets_this_attempt,
            cutoff=recovery_cutoff,
        )

        if attempt_collected_frames == 0:
            print(f"[{step_name}] Attempt {attempt} added 0 recovery frames "
                  f"(case_counts={aggregate_case_counts}); bailing to avoid infinite loop.")
            break

        remaining -= attempt_collected_frames
        attempt += 1
        worker_id_offset += spawns

    summary = {
        "dataset_step": str(step_name),
        "target_recovery_frames": target_recovery_frames,
        "case_counts": aggregate_case_counts,
        "policy_evaluated_count": aggregate_policy_evaluated,
        "policy_passed_count": aggregate_policy_pass,
        "policy_failed_count": aggregate_policy_failed,
        "policy_success_rate": (
            aggregate_policy_pass / aggregate_policy_evaluated
            if aggregate_policy_evaluated > 0 else 0.0
        ),
        "recovery_attempted_count": aggregate_recovery_attempted,
        "recovery_skipped_after_target_count": aggregate_recovery_skipped_after_target,
        "scenes_generated": aggregate_scenes,
        "collected_recovery_parquets": collected_parquets,
        "collected_recovery_records": collected_records,
    }
    with open(hunting_dir / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return collected_parquets, collected_records, summary


_RECOVERY_META_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


_RECOVERY_RGB_FEATURE_KEYS = ("observation.state", "action")


def clip_video_to_av1(src_video, dst_video, fps, frame_limit):
    src_video = Path(src_video)
    dst_video = Path(dst_video)
    dst_video.parent.mkdir(parents=True, exist_ok=True)

    options = {"g": "2", "crf": "18"}
    frames_written = 0
    with SuppressNativeStderr():
        with av.open(str(src_video), mode="r") as in_container:
            with av.open(str(dst_video), mode="w") as out_container:
                out_stream = None
                for frame_idx, frame in enumerate(in_container.decode(video=0)):
                    if frame_idx >= frame_limit:
                        break
                    bgr = frame.to_ndarray(format="bgr24")
                    if out_stream is None:
                        h, w = bgr.shape[:2]
                        out_stream = out_container.add_stream(
                            "libsvtav1",
                            rate=fps,
                            options=options,
                        )
                        out_stream.width = w
                        out_stream.height = h
                        out_stream.pix_fmt = "yuv420p"
                    av_frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
                    for packet in out_stream.encode(av_frame):
                        out_container.mux(packet)
                    frames_written += 1

                if out_stream is not None:
                    for packet in out_stream.encode():
                        out_container.mux(packet)

    if frames_written != frame_limit:
        raise ValueError(
            f"{src_video} yielded {frames_written} frames, expected {frame_limit}"
        )


def video_frame_count(video_path):
    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        return sum(1 for _ in container.decode(stream))


def copy_or_clip_video_to_av1(src_video, dst_video, fps, frame_limit):
    frame_count = video_frame_count(src_video)
    if frame_count == frame_limit:
        dst_video = Path(dst_video)
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_video, dst_video)
        return
    if frame_count < frame_limit:
        raise ValueError(
            f"{src_video} yielded {frame_count} frames, expected at least {frame_limit}"
        )
    clip_video_to_av1(src_video, dst_video, fps, frame_limit)


def rgb_recovery_records_from_dataset(dataset_dir, camera_keys):
    records = []
    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    episode_paths = sorted(dataset_dir.glob("meta/episodes/chunk-*/*.parquet"))
    if episode_paths:
        with open(info_path, "r") as f:
            info = json.load(f)
        episodes = pd.concat(
            [pd.read_parquet(path) for path in episode_paths],
            ignore_index=True,
        )
        generation = {int(row["episode"]): row
                      for row in json.loads((dataset_dir / "generation.info").read_text())}
        for _, episode in episodes.sort_values("episode_index").iterrows():
            data_path = dataset_dir / info["data_path"].format(
                chunk_index=int(episode["data/chunk_index"]),
                file_index=int(episode["data/file_index"]),
            )
            records.append({
                "recovery_parquet": str(data_path),
                "episode_index": int(episode["episode_index"]),
                "xml": generation[int(episode["episode_index"])]["xml"],
                "state_parquet": str(dataset_dir / "meta/state_buffers/chunk-000/file-000.parquet"),
                "camera_videos": {
                    camera_key: str(
                        dataset_dir
                        / info["video_path"].format(
                            video_key=camera_key,
                            chunk_index=int(episode[f"videos/{camera_key}/chunk_index"]),
                            file_index=int(episode[f"videos/{camera_key}/file_index"]),
                        )
                    )
                    for camera_key in camera_keys
                },
            })
        return records

    for parquet_path in sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet")):
        chunk_name = parquet_path.parent.name
        episode_name = parquet_path.with_suffix(".mp4").name
        records.append({
            "recovery_parquet": str(parquet_path),
            "camera_videos": {
                camera_key: str(
                    dataset_dir
                    / "videos"
                    / chunk_name
                    / camera_key
                    / episode_name
                )
                for camera_key in camera_keys
            },
        })
    return records


def compose_rgb_recovery_dataset(recovery_records, recovery_seconds,
                                 planner_lerobot_fps, out_dir,
                                 main_dataset_dir, camera_names):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "state_buffers" / "chunk-000").mkdir(parents=True, exist_ok=True)

    main_dir = Path(main_dataset_dir)
    with open(main_dir / "meta" / "info.json", "r") as f:
        main_info = json.load(f)

    camera_keys = [f"observation.images.{name}" for name in camera_names]
    recovery_features = {
        camera_key: main_info["features"][camera_key]
        for camera_key in camera_keys
    }
    for key in _RECOVERY_RGB_FEATURE_KEYS:
        recovery_features[key] = main_info["features"][key]
    recovery_features.update(_RECOVERY_META_FEATURES)

    shutil.copy2(main_dir / "meta" / "tasks.parquet", out_dir / "meta" / "tasks.parquet")
    stats_path = main_dir / "meta" / "stats.json"
    if stats_path.exists():
        shutil.copy2(stats_path, out_dir / "meta" / "stats.json")

    tasks_df = pd.read_parquet(out_dir / "meta" / "tasks.parquet")
    if "task" in tasks_df.columns:
        tasks_map = {
            int(row["task_index"]): str(row["task"])
            for _, row in tasks_df.iterrows()
        }
    else:
        tasks_map = {
            int(row["task_index"]): str(task)
            for task, row in tasks_df.iterrows()
        }
    default_task_idx = min(tasks_map)
    default_task = tasks_map[default_task_idx]

    info = copy.deepcopy(main_info)
    info["total_episodes"] = 0
    info["total_frames"] = 0
    info["total_tasks"] = len(tasks_map)
    info["total_videos"] = 0
    info["total_chunks"] = 0
    info["splits"] = {"train": "0:0"}
    info["data_path"] = main_info["data_path"]
    info["video_path"] = main_info["video_path"]
    info["features"] = recovery_features

    cutoff = int(round(recovery_seconds * planner_lerobot_fps))
    total_frames = 0
    ep_idx = 0
    keep_cols = set(_RECOVERY_RGB_FEATURE_KEYS)
    data_parts = []
    state_parts = []
    episode_rows = []
    generation_records = []

    for record in recovery_records:
        src_path = Path(record["recovery_parquet"])
        source_df = pd.read_parquet(src_path)
        if "episode_index" in record:
            source_df = source_df[
                source_df["episode_index"] == int(record["episode_index"])
            ].sort_values("frame_index")
        source_df = source_df.iloc[:cutoff].reset_index(drop=True)
        ep_length = len(source_df)
        if ep_length == 0:
            continue

        missing = [c for c in _RECOVERY_RGB_FEATURE_KEYS if c not in source_df.columns]
        if missing:
            raise KeyError(f"Recovery parquet {src_path} missing columns: {missing}")

        df = source_df[[c for c in source_df.columns if c in keep_cols]].copy()
        df["episode_index"] = np.int64(ep_idx)
        df["frame_index"] = np.arange(ep_length, dtype=np.int64)
        df["index"] = np.arange(total_frames, total_frames + ep_length, dtype=np.int64)
        df["timestamp"] = np.arange(ep_length, dtype=np.float32) / np.float32(info["fps"])
        df["task_index"] = np.int64(default_task_idx)
        data_parts.append(df)

        if "state_parquet" in record:
            recorded_states = pd.read_parquet(record["state_parquet"])
            episode_states = recorded_states[
                recorded_states["episode_index"] == int(record["episode_index"])
            ].sort_values("frame_index").iloc[:cutoff]
        else:
            episode_states = source_df
        if len(episode_states) != ep_length:
            raise ValueError(f"Recovery state/frame counts differ for {src_path}")
        state_parts.append(pd.DataFrame({
            "episode_index": np.full(ep_length, ep_idx, dtype=np.int64),
            "frame_index": np.arange(ep_length, dtype=np.int64),
            "timestamp": df["timestamp"].to_numpy(dtype=np.float32),
            "state_buf": list(episode_states["state_buf"].values),
            "last_gripper_status": list(episode_states["last_gripper_status"].values),
        }))

        episode_row = {
            "episode_index": ep_idx,
            "tasks": [default_task],
            "length": ep_length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + ep_length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for camera_key in camera_keys:
            dst_video = (
                out_dir
                / "videos"
                / camera_key
                / "chunk-000"
                / f"file-{ep_idx:03d}.mp4"
            )
            copy_or_clip_video_to_av1(
                src_video=record["camera_videos"][camera_key],
                dst_video=dst_video,
                fps=int(info["fps"]),
                frame_limit=ep_length,
            )
            episode_row[f"videos/{camera_key}/chunk_index"] = 0
            episode_row[f"videos/{camera_key}/file_index"] = ep_idx
            episode_row[f"videos/{camera_key}/from_timestamp"] = 0.0
            episode_row[f"videos/{camera_key}/to_timestamp"] = ep_length / float(info["fps"])

        episode_rows.append(episode_row)
        generation_records.append({
            "episode": ep_idx,
            "xml": record["xml"],
            "source_recovery_parquet": str(src_path),
            "source_episode": record.get("episode_index"),
            "success": True,
        })
        total_frames += ep_length
        ep_idx += 1

    data = pd.concat(data_parts, ignore_index=True) if data_parts else pd.DataFrame()
    data.to_parquet(out_dir / "data" / "chunk-000" / "file-000.parquet", index=False)
    if state_parts:
        states = pd.concat(state_parts, ignore_index=True)
        states.to_parquet(
            out_dir / "meta" / "state_buffers" / "chunk-000" / "file-000.parquet",
            index=False,
        )
    pd.DataFrame(episode_rows).to_parquet(
        out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        index=False,
    )

    info["total_episodes"] = ep_idx
    info["total_frames"] = total_frames
    info["total_chunks"] = 1 if ep_idx > 0 else 0
    info["total_videos"] = ep_idx * len(camera_keys)
    info["splits"] = {"train": f"0:{ep_idx}"}
    if ep_idx > 0:
        from lerobot.datasets.video_utils import get_video_info

        for camera_key in camera_keys:
            video_path = (
                out_dir / "videos" / camera_key / "chunk-000" / "file-000.mp4"
            )
            info["features"][camera_key]["info"] = get_video_info(video_path)
    with open(out_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(out_dir / "generation.info", "w") as f:
        json.dump(generation_records, f, indent=4)

    return total_frames


def _find_prev_recovery_dataset(dataset_prefix, current_step, recovery_seconds):
    rs_str = _fmt_float_for_path(recovery_seconds)
    pattern = re.compile(
        rf"^dataset_step_(\d+)_X{re.escape(rs_str)}s_recovery$"
    )

    candidates = []
    for d in Path(dataset_prefix).glob(f"dataset_step_*_X{rs_str}s_recovery"):
        p = Path(d)
        match = pattern.match(p.name)
        if match:
            candidates.append((int(match.group(1)), p))

    candidates = [c for c in candidates if c[0] < current_step]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[-1][1]


def _find_prev_ckpt(ckpt_prefix, current_step, start_ckpt_final):
    candidates = []
    pattern = re.compile(r"^dataset_step_(\d+)$")
    for d in Path(ckpt_prefix).glob("dataset_step_*"):
        p = Path(d)
        if not p.is_dir():
            continue
        match = pattern.match(p.name)
        if not match:
            continue
        ckpt_step = int(match.group(1))
        final_path = saved_checkpoint_path(p)
        if final_path is not None:
            candidates.append((ckpt_step, final_path))

    candidates = [c for c in candidates if c[0] < current_step]
    if not candidates:
        start_ckpt = saved_checkpoint_path(start_ckpt_final.parent)
        if start_ckpt is not None:
            return start_ckpt
        raise FileNotFoundError(
            f"No saved checkpoint found before {dataset_step_name(current_step)} "
            f"under {ckpt_prefix}; start checkpoint has no model checkpoint under "
            f"{start_ckpt_final.parent}"
        )
    candidates.sort(key=lambda c: c[0])
    return candidates[-1][1]


def saved_checkpoint_path(ckpt_root):
    ckpt_root = Path(ckpt_root)
    final_final = ckpt_root / "final" / "final"
    if (final_final / "model.safetensors").exists():
        return final_final
    if (final_final / "params").exists():
        return final_final
    final_path = ckpt_root / "final"
    if (final_path / "model.safetensors").exists():
        return final_path
    if (final_path / "params").exists():
        return final_path
    step_pattern = re.compile(r"^step_(\d+)$")
    step_candidates = []
    for parent in [ckpt_root / "final", ckpt_root]:
        for step_dir in parent.glob("step_*"):
            match = step_pattern.match(step_dir.name)
            if match and (step_dir / "model.safetensors").exists():
                step_candidates.append((int(match.group(1)), step_dir))
    if step_candidates:
        step_candidates.sort(key=lambda c: c[0])
        return step_candidates[-1][1]
    openpi_candidates = []
    for parent in [ckpt_root / "final", ckpt_root]:
        for step_dir in parent.glob("*"):
            if step_dir.is_dir() and step_dir.name.isdigit() and (step_dir / "params").exists():
                openpi_candidates.append((int(step_dir.name), step_dir))
    if openpi_candidates:
        openpi_candidates.sort(key=lambda c: c[0])
        return openpi_candidates[-1][1]
    return None


def _replay_eval_state(rollout_prefix, continue_dataset_step_number, success_rate_threshold):
    """Reconstruct (eval_every_iter, best_eval_success_rate, rollouts_since_best)
    from existing dataset_step_*_test_rollout/summary.json files.

    Without this, restarting the script resets these in-memory flags and loses
    the high-success-mode switch and early-stop counter. Only summaries
    strictly before `continue_dataset_step_number` are replayed, matching what
    the loop would have observed.
    """
    pattern = re.compile(r"^dataset_step_(\d+)_test_rollout$")
    entries = []
    for d in Path(rollout_prefix).glob("dataset_step_*_test_rollout"):
        p = Path(d)
        m = pattern.match(p.name)
        if not m:
            continue
        summary_path = p / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r") as f:
            summary = json.load(f)
        sr = summary["success_rate"]
        key = int(m.group(1))
        if key >= continue_dataset_step_number:
            continue
        entries.append((key, float(sr)))

    entries.sort(key=lambda x: x[0])

    eval_every_iter = False
    best = None
    rollouts_since_best = 0
    for _, sr in entries:
        if sr > success_rate_threshold:
            eval_every_iter = True
        if eval_every_iter:
            if best is None or sr > best:
                best = sr
                rollouts_since_best = 1
            else:
                rollouts_since_best += 1
    return eval_every_iter, best, rollouts_since_best, entries


def target_step_numbers(
    start_dataset_step_number,
    continue_dataset_step_number,
    total_dataset_step_number,
    incremental_dataset_step_number,
):
    if incremental_dataset_step_number <= 0:
        raise ValueError("--incremental_dataset_step_number must be > 0")
    if start_dataset_step_number <= 0:
        raise ValueError("--start_dataset_step_number must be > 0")
    if total_dataset_step_number <= start_dataset_step_number:
        raise ValueError(
            f"--total_dataset_step_number ({total_dataset_step_number}) must be > start "
            f"dataset step ({start_dataset_step_number})"
        )
    if (total_dataset_step_number - start_dataset_step_number) % incremental_dataset_step_number:
        raise ValueError(
            "--total_dataset_step_number must align with --start_dataset_step_number "
            "and --incremental_dataset_step_number"
        )

    first_step = start_dataset_step_number + incremental_dataset_step_number
    if continue_dataset_step_number is not None:
        if continue_dataset_step_number < first_step:
            raise ValueError(
                f"--continue_dataset_step_number ({continue_dataset_step_number}) "
                f"must be >= first active-learning step ({first_step})"
            )
        if continue_dataset_step_number > total_dataset_step_number:
            raise ValueError(
                f"--continue_dataset_step_number ({continue_dataset_step_number}) "
                f"must be <= --total_dataset_step_number ({total_dataset_step_number})"
            )
        if (
            continue_dataset_step_number - start_dataset_step_number
        ) % incremental_dataset_step_number:
            raise ValueError(
                "--continue_dataset_step_number must align with "
                "--start_dataset_step_number and --incremental_dataset_step_number"
            )
        first_step = continue_dataset_step_number

    return list(range(first_step, total_dataset_step_number + 1, incremental_dataset_step_number))


def openpi_fsdp_devices() -> int:
    return int(os.environ.get("OPENPI_FSDP_DEVICES", "2"))


def training_command(
    *,
    python_executable,
    representation,
    policy_name,
    dataset_dir,
    recovery_dataset_dir,
    new_recovery_episode_count,
    policy_config_path,
    tasks_config_path,
    pretrained_ckpt_path,
    ckpt_save_dir,
    device,
    seed,
    task_name,
    use_relative_action,
    no_ee,
    add_action_noise,
):
    if policy_name == "Pi05":
        cmd = [
            python_executable, "-m",
            "policy.policy_training.rgb.train.Pi05",
            "--dataset_dir", str(dataset_dir),
            "--policy_config_path", str(policy_config_path),
            "--tasks_config_path", str(tasks_config_path),
            "--pretrained_ckpt_path", str(pretrained_ckpt_path),
            "--ckpt_save_dir", str(ckpt_save_dir),
            "--device", device,
            "--seed", str(seed),
            "--task", task_name,
            "--fsdp-devices", str(openpi_fsdp_devices()),
        ]
        save_interval = os.environ.get("OPENPI_SAVE_INTERVAL_STEPS")
        if save_interval is None:
            cmd.append("--no-save_intermediate_ckpts")
        else:
            cmd.extend(["--save_interval", save_interval])
        if recovery_dataset_dir is not None and recovery_dataset_dir.exists():
            cmd.extend(["--recovery_dataset_path", str(recovery_dataset_dir)])
            cmd.extend([
                "--new_recovery_episode_count",
                str(new_recovery_episode_count),
            ])
    else:
        cmd = [
            python_executable, "-m",
            f"policy.policy_training.{representation}.train.{policy_name}_combined_v3",
            "--dataset_dir", str(dataset_dir),
        ]
        if recovery_dataset_dir is not None and recovery_dataset_dir.exists():
            cmd.extend(["--recovery_dataset_path", str(recovery_dataset_dir)])
        if representation == "rgb" and policy_name in {
            "ACT",
            "diffusion_encoder_v2",
        }:
            cmd.extend([
                "--new_recovery_episode_count",
                str(new_recovery_episode_count),
            ])
        cmd.extend([
            "--policy_config_path", str(policy_config_path),
            "--tasks_config_path", str(tasks_config_path),
            "--pretrained_ckpt_path", str(pretrained_ckpt_path),
            "--ckpt_save_dir", str(ckpt_save_dir),
            "--device", device,
            "--seed", str(seed),
            "--task", task_name,
        ])
    if not use_relative_action:
        cmd.append("--use_absolute_action")
    if no_ee:
        cmd.append("--no_ee")
    if add_action_noise:
        cmd.append("--add_action_noise")
    return cmd


def run_test_rollout(
    step_name,
    test_scene_paths,
    devices,
    mujoco_egl_device_ids,
    rollout_prefix,
    ckpt_to_eval,
    policy_cfg_path,
    policy_name,
    representation,
    eval_module_name,
    task_name,
    tasks_config_path,
    python_executable,
    log_video,
    use_relative_action,
    rollout_batch_size,
    no_ee=False,
):
    print(f"\n[{step_name} - Eval] Rolling out on {len(test_scene_paths)} test scenes "
          f"across {len(devices)} GPUs...")
    rollout_out_dir = Path(rollout_prefix) / f"{step_name}_test_rollout"
    rollout_out_dir.mkdir(parents=True, exist_ok=True)

    test_chunk_size = math.ceil(len(test_scene_paths) / len(devices))
    scene_chunks = [
        test_scene_paths[i:i + test_chunk_size]
        for i in range(0, len(test_scene_paths), test_chunk_size)
    ]

    processes = []
    chunk_dirs = []
    for chunk_idx, scene_chunk in enumerate(scene_chunks):
        if not scene_chunk:
            continue
        chunk_gpu = devices[chunk_idx % len(devices)]
        chunk_dir = rollout_out_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_dirs.append(chunk_dir)

        cmd_eval = [
            python_executable, "-m",
            f"policy.policy_evaluation.{eval_module_name}",
            "--xml_list", *scene_chunk,
            "--output_dir", str(chunk_dir),
            "--pretrained_ckpt_path", str(ckpt_to_eval),
            "--policy_config_path", str(policy_cfg_path),
            "--device", chunk_gpu,
            "--task", task_name,
            "--tasks_config_path", tasks_config_path,
        ]
        if not use_relative_action:
            cmd_eval.append("--use_absolute_action")
        cmd_eval.extend(
            [
                "--rollout_batch_size",
                str(rollout_batch_size),
                "--multiprocess_envs",
            ]
        )
        if log_video:
            cmd_eval.append("--log_video")
        if no_ee:
            cmd_eval.append("--no_ee")
        out_file = open(str(chunk_dir / "eval.out"), "w")
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["MUJOCO_EGL_DEVICE_ID"] = mujoco_egl_device_ids[chunk_gpu]
        p = subprocess.Popen(cmd_eval, env=env, stdout=out_file, stderr=subprocess.STDOUT)
        processes.append((chunk_idx, p, out_file))

    all_ok = True
    for chunk_idx, p, out_file in processes:
        p.wait()
        out_file.close()
        gc.collect()
        if p.returncode != 0:
            print(f"[{step_name} - Eval] Chunk {chunk_idx} failed with code {p.returncode}.")
            all_ok = False
    if not all_ok:
        raise RuntimeError(f"[{step_name} - Eval] One or more eval chunks failed.")

    aggregated = []
    total_evals = 0
    successes = 0
    for chunk_dir in chunk_dirs:
        info_path = chunk_dir / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing eval info.json for {chunk_dir}")
        with open(info_path, "r") as f:
            data = json.load(f)
        aggregated.extend(data)
        for item in data:
            total_evals += 1
            if item["evaluation"]["status"] == "success":
                successes += 1

    with open(rollout_out_dir / "info.json", "w") as f:
        json.dump(aggregated, f, indent=4)
    success_rate = (successes / total_evals) if total_evals > 0 else 0
    print(f"[{step_name} - Eval] Success rate: "
          f"{successes}/{total_evals} ({success_rate*100:.2f}%)")
    with open(rollout_out_dir / "summary.json", "w") as f:
        json.dump({
            "dataset_step": step_name,
            "total_evals": total_evals,
            "successes": successes,
            "success_rate": success_rate,
        }, f, indent=4)

    return success_rate, successes, total_evals


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning corrective (incremental recovery-data driven)."
    )
    parser.add_argument(
        "--policy",
        choices=SUPPORTED_POLICIES,
        type=str,
        required=True,
        help="Policy to train and evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Space-separated devices (e.g., 'cuda:0 cuda:1')",
    )
    parser.add_argument("--policy_preset", default="default", help="Named preset in the policy YAML")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--representation", type=str, choices=("rgb",), default="rgb",
                        help="Must be rgb")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--scene_dir", type=Path, help="Task scene directory containing evaluation splits")
    parser.add_argument("--test_scene_count", type=int, default=100)
    parser.add_argument("--evaluate_every_round", action="store_true")
    parser.add_argument("--training_workers", type=int, default=4)
    parser.add_argument("--corrective_config", default="configurations/corrective.yaml")
    parser.add_argument("--pos_diff_threshold", type=float, default=None)
    parser.add_argument("--recovery_seconds", type=float, default=None,
                        help="X seconds of recovery taken per scene; suffix on "
                             "recovery dataset dir")

    parser.add_argument("--difficulty", type=str, default="uniform")
    parser.add_argument("--incremental_dataset_step_number", type=int, default=1000,
                        help="Frame-step increment between active-learning dataset targets.")
    parser.add_argument("--start_dataset_step_number", type=int, default=None,
                        help="Start dense dataset/checkpoint frame-step target. "
                             "Omit it to resume without copying dense_sample "
                             "dataset/checkpoint.")
    parser.add_argument("--continue_dataset_step_number", type=int, default=None,
                        help="First active-learning dataset-step target to process when resuming.")
    parser.add_argument("--total_dataset_step_number", type=int, required=True,
                        help="Final cumulative frame-step target, inclusive.")
    parser.add_argument("--evaluate_dataset_step_number", type=int, default=None,
                        help="Evaluate only when the dataset-step target is a "
                             "multiple of this value. Defaults to "
                             "--incremental_dataset_step_number.")
    parser.add_argument("--rollout_batch_size", type=int, default=50,
                        help="Number of XMLs to roll out per eval policy batch.")
    parser.add_argument("--incremental_training_steps", type=int, required=True,
                        help="Fine-tuning steps for each active-learning dataset step.")
    parser.add_argument("--success_rate_threshold", type=float,
                        default=DEFAULT_SUCCESS_RATE_THRESHOLD,
                        help="Success-rate threshold for entering "
                             "high-success-regime early stopping. Default: 0.95.")
    parser.add_argument("--disable_early_stopping", action="store_true",
                        help="Continue to the final dataset target regardless of evaluation success.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_scenes_per_worker", type=int, default=1000,
                        help="Per-worker safety cap on scene generations per attempt.")
    parser.add_argument("--not_log_video", action="store_true",
                        help="Disable policy-rollout video logging (default: log video).")
    parser.add_argument("--no_ee", action="store_true",
                        help="Zero observation.state during policy training and rollout.")
    parser.add_argument("--add_action_noise", action="store_true",
                        help="Enable Jacobian-based TCP noise injection during training.")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip test-scene eval rollouts.")
    parser.add_argument("--only_evaluate", action="store_true",
                        help="Only evaluate the checkpoint for --continue_dataset_step_number, "
                             "or --total_dataset_step_number when continue is not set.")

    args = parser.parse_args()
    with open(args.corrective_config) as stream:
        corrective_config = yaml.safe_load(stream)
    recovery_seconds = args.recovery_seconds if args.recovery_seconds is not None else float(corrective_config["recovery_seconds"])
    pos_diff_threshold = args.pos_diff_threshold if args.pos_diff_threshold is not None else float(corrective_config["pos_diff_threshold"])
    if pos_diff_threshold <= 0:
        raise ValueError("--pos_diff_threshold must be positive")

    if recovery_seconds < 1.6:
        raise ValueError(
            f"--recovery_seconds must be > 1.5 (got {recovery_seconds})."
        )
    if args.incremental_dataset_step_number <= 0:
        raise ValueError(
            "--incremental_dataset_step_number must be > 0 "
            f"(got {args.incremental_dataset_step_number})."
        )
    if args.start_dataset_step_number is None:
        assert args.continue_dataset_step_number is not None, (
            "--continue_dataset_step_number is required when "
            "--start_dataset_step_number is omitted."
        )
    elif args.start_dataset_step_number <= 0:
        raise ValueError("--start_dataset_step_number must be > 0")
    if args.incremental_training_steps <= 0:
        raise ValueError("--incremental_training_steps must be > 0")
    evaluate_dataset_step_number = (
        args.incremental_dataset_step_number
        if args.evaluate_dataset_step_number is None
        else args.evaluate_dataset_step_number
    )
    if evaluate_dataset_step_number <= 0:
        raise ValueError("--evaluate_dataset_step_number must be > 0")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout_batch_size must be > 0")
    if not 0 <= args.success_rate_threshold <= 1:
        raise ValueError(
            "--success_rate_threshold must be between 0 and 1 "
            f"(got {args.success_rate_threshold})."
        )

    policy_runtime = read_policy_runtime(args.policy, args.policy_preset)
    if policy_runtime.use_relative_action:
        raise ValueError("Corrective collection requires absolute actions")
    planner_lerobot_fps = policy_runtime.lerobot_fps
    policy_lerobot_fps = policy_runtime.lerobot_fps
    policy_infer_freq = policy_runtime.policy_freq

    policy2planner_factor = planner_lerobot_fps // policy_lerobot_fps
    recovery_cutoff = int(round(recovery_seconds * planner_lerobot_fps))
    print(
        f"Policy config: chunk_size={policy_runtime.chunk_size}, "
        f"policy_freq={policy_runtime.policy_freq}, "
        f"lerobot_fps={policy_runtime.lerobot_fps}, "
        f"use_relative_action={policy_runtime.use_relative_action}. "
        "Dataset steps are counted as parquet frames. "
        f"Fine-tuning runs for {args.incremental_training_steps} steps "
        "per active-learning dataset step."
    )
    print(
        f"planner_lerobot_fps={planner_lerobot_fps}, "
        f"policy_lerobot_fps={policy_lerobot_fps}, "
        f"policy_infer_freq={policy_infer_freq}, "
        f"policy2planner_factor={policy2planner_factor}, "
        f"recovery_cutoff={recovery_cutoff} frames"
    )

    policy_name = args.policy
    policy_run_name = policy_name if args.policy_preset == "default" else f"{policy_name}_{args.policy_preset}"
    eval_module = f"{policy_name}_ensemble"
    task_name = args.task
    representation = args.representation
    devices = args.device.split()
    if not devices:
        raise ValueError("--device must contain at least one device.")
    device = devices[0]
    mujoco_egl_device_ids = {device: egl_device(device) for device in devices}
    seed = args.seed
    np.random.seed(seed)

    cache_root = Path(os.environ.get("CODEGEN_CACHE_DIR", Path.home() / ".cache"))
    cache_env = cache_environment(cache_root)
    for key, path in cache_env.items():
        if key != "WANDB_MODE":
            Path(path).mkdir(parents=True, exist_ok=True)
    os.environ.update(cache_env)

    with open("configurations/tasks.yaml", "r") as f:
        task_config = yaml.safe_load(f)
    if task_name not in task_config["tasks"]:
        raise ValueError(f"Task '{task_name}' not found in configurations/tasks.yaml")
    task_cfg = task_config["tasks"][task_name]
    camera_names = task_cfg["camera_names"]
    camera_keys = [f"observation.images.{name}" for name in camera_names]

    workspace_segment = (
        f"{task_name}"
        f"_pos_{_fmt_float_for_path(pos_diff_threshold)}"
        f"_time_{_fmt_float_for_path(recovery_seconds)}"
    )

    policy_tolerance = task_cfg["policy_tolerance"]
    start_dataset_step_number = args.start_dataset_step_number

    # --- paths --------------------------------------------------------------
    rep_dir = f"{representation}_no_ee" if args.no_ee else representation
    if not policy_runtime.use_relative_action:
        rep_dir = f"{rep_dir}_abs_action"
    if args.add_action_noise:
        rep_dir = f"{rep_dir}_add_action_noise"

    workspace_root = (
        Path(args.save_dir)
        / task_name
        / WORKSPACE_TAG
        / workspace_segment
        / rep_dir
        / f"{policy_run_name}_seed_{seed}"
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    policy_cfg_path = workspace_root / "policy.yaml"
    with policy_cfg_path.open("w") as stream:
        yaml.safe_dump(policy_runtime.config, stream, sort_keys=False)
    dataset_prefix = workspace_root / "lerobot_dataset"
    ckpt_prefix = workspace_root / "checkpoints"
    rollout_prefix = workspace_root / "rollouts"

    if args.only_evaluate:
        eval_step = args.continue_dataset_step_number or args.total_dataset_step_number
        if eval_step <= 0:
            raise ValueError("--only_evaluate requires a positive dataset-step target")
        step_numbers = [eval_step]
        start_step_name = None
        workspace_start_dataset = None
        workspace_start_ckpt = None
    elif start_dataset_step_number is None:
        start_dataset_pattern = re.compile(r"^dataset_step_(\d+)$")
        start_dataset_candidates = []
        for dataset_dir in dataset_prefix.glob("dataset_step_*"):
            match = start_dataset_pattern.match(dataset_dir.name)
            if match and (dataset_dir / "data").exists():
                start_dataset_candidates.append((int(match.group(1)), dataset_dir))
        if len(start_dataset_candidates) != 1:
            raise ValueError(
                "Resume mode expected exactly one existing workspace start dataset "
                f"under {dataset_prefix}, found {len(start_dataset_candidates)}."
            )
        start_dataset_step_number, workspace_start_dataset = start_dataset_candidates[0]
        start_step_name = dataset_step_name(start_dataset_step_number)
        workspace_start_ckpt = ckpt_prefix / start_step_name
        step_numbers = target_step_numbers(
            start_dataset_step_number,
            args.continue_dataset_step_number,
            args.total_dataset_step_number,
            args.incremental_dataset_step_number,
        )
    else:
        start_step_name = dataset_step_name(start_dataset_step_number)
        workspace_start_dataset = dataset_prefix / start_step_name
        workspace_start_ckpt = ckpt_prefix / start_step_name
        step_numbers = target_step_numbers(
            start_dataset_step_number,
            args.continue_dataset_step_number,
            args.total_dataset_step_number,
            args.incremental_dataset_step_number,
        )

    tasks_config_path = "configurations/tasks.yaml"

    python_executable = sys.executable

    print("================ INITIALIZATION ================")
    print(f"workspace_root={workspace_root}")
    print(
        f"start_dataset_step_number={start_dataset_step_number}, "
        f"incremental_dataset_step_number={args.incremental_dataset_step_number}, "
        f"total_dataset_step_number={args.total_dataset_step_number}, "
        f"evaluate_dataset_step_number={evaluate_dataset_step_number}"
    )
    print(
        f"Running dataset-step targets {step_numbers[0]}..{step_numbers[-1]} "
        f"in increments of {args.incremental_dataset_step_number} frames."
    )
    print(f"use_relative_action={policy_runtime.use_relative_action}, use_ee={not args.no_ee}")

    # --- test scenes for eval ----------------------------------------------
    task_scene_dir = args.scene_dir or Path("data/scenes") / task_name
    test_scene_dir = task_scene_dir / f"test_{args.difficulty}"
    test_scene_paths = []
    if not args.skip_evaluation:
        if args.difficulty == "uniform":
            test_scene_paths = scene_paths(task_name, test_scene_dir, 0, args.test_scene_count)
        else:
            test_scene_paths = [str(test_scene_dir / f"sample_{i}/tmp_scene.xml") for i in range(args.test_scene_count)]
            if not all(Path(path).is_file() for path in test_scene_paths):
                raise FileNotFoundError(f"Expected {args.test_scene_count} evaluation scenes under {test_scene_dir}")

    if args.only_evaluate:
        eval_step = args.continue_dataset_step_number or args.total_dataset_step_number
        step_name = dataset_step_name(eval_step)
        ckpt_root = ckpt_prefix / step_name
        ckpt_to_eval = saved_checkpoint_path(ckpt_root)
        if ckpt_to_eval is None:
            raise FileNotFoundError(
                f"--only_evaluate expected a saved checkpoint for {step_name} "
                f"under {ckpt_root}"
            )
        if not test_scene_paths:
            raise FileNotFoundError(
                f"--only_evaluate found no test scenes at {test_scene_dir}"
            )
        rollout_prefix.mkdir(parents=True, exist_ok=True)
        print(f"[{step_name}] --only_evaluate set; evaluating checkpoint: {ckpt_to_eval}")
        run_test_rollout(
            step_name=step_name,
            test_scene_paths=test_scene_paths,
            devices=devices,
            mujoco_egl_device_ids=mujoco_egl_device_ids,
            rollout_prefix=rollout_prefix,
            ckpt_to_eval=ckpt_to_eval,
            policy_cfg_path=policy_cfg_path,
            policy_name=policy_name,
            representation=representation,
            eval_module_name=eval_module,
            task_name=task_name,
            tasks_config_path=tasks_config_path,
            python_executable=python_executable,
            log_video=not args.not_log_video,
            use_relative_action=policy_runtime.use_relative_action,
            rollout_batch_size=args.rollout_batch_size,
            no_ee=args.no_ee,
        )
        return

    dataset_prefix.mkdir(parents=True, exist_ok=True)
    ckpt_prefix.mkdir(parents=True, exist_ok=True)
    rollout_prefix.mkdir(parents=True, exist_ok=True)

    if args.start_dataset_step_number is None:
        if not (workspace_start_dataset / "data").exists():
            raise FileNotFoundError(
                f"Workspace start dataset not found: {workspace_start_dataset}"
            )
        print(f"Using existing {start_step_name} dataset at {workspace_start_dataset}")
    else:
        start_dense_dataset = (
            Path(args.save_dir)
            / task_name
            / "dense_sample"
            / "lerobot_dataset"
            / start_step_name
        )
        start_dense_ckpt = (
            Path(args.save_dir)
            / task_name
            / "dense_sample"
            / rep_dir
            / start_step_name
            / f"{policy_run_name}_ckpt_seed_{seed}"
        )
        if not (start_dense_dataset / "data").exists():
            raise FileNotFoundError(f"Start dense dataset not found: {start_dense_dataset}")
        if not (start_dense_ckpt / "final").exists():
            raise FileNotFoundError(f"Start dense checkpoint not found: {start_dense_ckpt}")

        if not (workspace_start_dataset / "data").exists():
            if workspace_start_dataset.exists():
                shutil.rmtree(workspace_start_dataset)
            shutil.copytree(start_dense_dataset, workspace_start_dataset)
            print(f"Copied {start_step_name} dataset -> {workspace_start_dataset}")

        if not (workspace_start_ckpt / "final").exists():
            if workspace_start_ckpt.exists():
                shutil.rmtree(workspace_start_ckpt)
            shutil.copytree(start_dense_ckpt, workspace_start_ckpt)
            print(f"Copied {start_step_name} checkpoint -> {workspace_start_ckpt}")

    start_dataset_frames = dataset_frame_count(workspace_start_dataset)
    if start_dataset_frames <= 0:
        raise ValueError(f"{start_step_name} dataset is empty: {workspace_start_dataset}")
    print(f"{start_step_name} dataset frames: {start_dataset_frames}")

    start_ckpt_final = workspace_start_ckpt / "final"
    eval_every_iter, best_eval_success_rate, rollouts_since_best, replayed_entries = \
        _replay_eval_state(rollout_prefix, step_numbers[0], args.success_rate_threshold)
    if replayed_entries:
        best_str = (
            f"{best_eval_success_rate * 100:.2f}%"
            if best_eval_success_rate is not None else "None"
        )
        print(
            f"Replayed eval state from {len(replayed_entries)} prior test_rollout(s) "
            f"before {dataset_step_name(step_numbers[0])}: "
            f"eval_every_iter={eval_every_iter}, "
            f"best_eval_success_rate={best_str}, "
            f"rollouts_since_best={rollouts_since_best}"
        )

    for target_step in step_numbers:
        step_name = dataset_step_name(target_step)
        print(f"\n================ {step_name} ================")

        prev_dataset_dir = _find_prev_recovery_dataset(
            dataset_prefix,
            target_step,
            recovery_seconds,
        )
        if prev_dataset_dir is not None:
            prior_recovery_parquets = dataset_parquet_paths(prev_dataset_dir)
            prior_recovery_frames = dataset_frame_count(prev_dataset_dir)
            prior_rgb_records = (
                rgb_recovery_records_from_dataset(prev_dataset_dir, camera_keys)
                if representation == "rgb"
                else []
            )
        else:
            prior_recovery_parquets = []
            prior_rgb_records = []
            prior_recovery_frames = 0

        current_total_before_step = start_dataset_frames + prior_recovery_frames
        target_recovery_frames = max(
            target_step - current_total_before_step,
            0,
        )
        print(
            f"[{step_name}] target_step={target_step}, "
            f"current_total_before_step={current_total_before_step}, "
            f"prior_recovery_frames={prior_recovery_frames} "
            f"(from {len(prior_recovery_parquets)} prior parquets), "
            f"target_recovery_frames={target_recovery_frames}"
        )

        ckpt_to_finetune = _find_prev_ckpt(
            ckpt_prefix,
            target_step,
            start_ckpt_final,
        )
        if not ckpt_to_finetune.exists():
            raise FileNotFoundError(
                f"Resolved prior checkpoint does not exist: {ckpt_to_finetune}"
            )
        print(f"[{step_name}] Finetuning from checkpoint: {ckpt_to_finetune}")

        recovery_dataset_dir = None
        new_collected = []
        new_rgb_records = []
        candidate = (
            dataset_prefix
            / f"{step_name}_X{_fmt_float_for_path(recovery_seconds)}s_recovery"
        )

        if (candidate / "meta" / "info.json").exists():
            recovered_records = rgb_recovery_records_from_dataset(candidate, camera_keys)
            recovery_dataset_dir = candidate
            new_rgb_records = recovered_records[len(prior_rgb_records):]
            print(f"[{step_name}] Reusing completed recovery dataset: {candidate}")
        else:
            if target_recovery_frames > 0:
                new_collected, new_rgb_records, summary = collect_recovery_data(
                    step_name=step_name,
                    target_recovery_frames=target_recovery_frames,
                    devices=devices,
                    mujoco_egl_device_ids=mujoco_egl_device_ids,
                    rollout_dir=rollout_prefix,
                    ckpt_path=ckpt_to_finetune,
                    policy_cfg_path=policy_cfg_path,
                    representation=representation,
                    policy_name=policy_name,
                    task=task_name,
                    tasks_config_path=tasks_config_path,
                    pos_diff_threshold=pos_diff_threshold,
                    policy_tolerance=policy_tolerance,
                    planner_lerobot_fps=planner_lerobot_fps,
                    policy_lerobot_fps=policy_lerobot_fps,
                    policy_infer_freq=policy_infer_freq,
                    recovery_seconds=recovery_seconds,
                    recovery_cutoff=recovery_cutoff,
                    python_executable=python_executable,
                    seed=seed + target_step,
                    max_scenes_per_worker=args.max_scenes_per_worker,
                    log_video=not args.not_log_video,
                    rollout_batch_size=args.rollout_batch_size,
                    no_ee=args.no_ee,
                )
                print(
                    f"[{step_name}] worker aggregate: {summary['case_counts']}, "
                    f"scenes={summary['scenes_generated']}, "
                    f"policy_passed={summary['policy_passed_count']}/"
                    f"{summary['policy_evaluated_count']} "
                    f"({summary['policy_success_rate'] * 100:.2f}%), "
                    f"recovery_attempted={summary['recovery_attempted_count']}, "
                    f"recovery_skipped_after_target="
                    f"{summary['recovery_skipped_after_target_count']}, "
                    f"new_recovery_parquets={len(new_collected)}"
                )
            else:
                print(f"[{step_name}] No frame gap to fill; reusing prior recovery data.")

            all_rgb_records = prior_rgb_records + new_rgb_records
            if all_rgb_records:
                rows = compose_rgb_recovery_dataset(
                    all_rgb_records,
                    recovery_seconds,
                    planner_lerobot_fps,
                    candidate,
                    main_dataset_dir=workspace_start_dataset,
                    camera_names=camera_names,
                )
                print(
                    f"[{step_name}] composed cumulative RGB recovery dataset: "
                    f"{candidate} ({rows} frames; "
                    f"{len(prior_rgb_records)} prior + "
                    f"{len(new_rgb_records)} new episodes)"
                )
                if rows > 0:
                    recovery_dataset_dir = candidate

        if recovery_dataset_dir is not None and recovery_dataset_dir.exists():
            current_recovery_frames = dataset_frame_count(recovery_dataset_dir)
        else:
            current_recovery_frames = prior_recovery_frames
        current_total_steps = start_dataset_frames + current_recovery_frames
        print(
            f"[{step_name}] current_total_steps={current_total_steps} "
            f"({start_step_name}={start_dataset_frames}, "
            f"recovery={current_recovery_frames})"
        )

        print(f"[{step_name}] Training combined policy...")
        save_ckpt_dir = ckpt_prefix / step_name / "final"
        save_ckpt_dir.mkdir(parents=True, exist_ok=True)

        training_steps = args.incremental_training_steps
        temp_cfg_path = workspace_root / f"config_{step_name}.yaml"
        iter_cfg = copy.deepcopy(policy_runtime.config)
        iter_cfg[policy_runtime.policy_key]["setting"]["repo_id"] = task_name
        iter_cfg[policy_runtime.policy_key]["train"]["num_steps_finetune"] = training_steps
        iter_cfg[policy_runtime.policy_key]["train"]["num_workers_rgb"] = args.training_workers
        iter_cfg[policy_runtime.policy_key]["train"][
            f"num_steps_{representation}"
        ] = training_steps
        with open(temp_cfg_path, "w") as f:
            yaml.safe_dump(iter_cfg, f)

        cmd_train = training_command(
            python_executable=python_executable,
            representation=representation,
            policy_name=policy_name,
            dataset_dir=workspace_start_dataset,
            recovery_dataset_dir=recovery_dataset_dir,
            new_recovery_episode_count=len(new_rgb_records),
            policy_config_path=temp_cfg_path,
            tasks_config_path=tasks_config_path,
            pretrained_ckpt_path=ckpt_to_finetune,
            ckpt_save_dir=save_ckpt_dir,
            device=device,
            seed=seed,
            task_name=task_name,
            use_relative_action=policy_runtime.use_relative_action,
            no_ee=args.no_ee,
            add_action_noise=args.add_action_noise,
        )
        try:
            run_command(
                cmd_train,
                cache_env=cache_env,
                env={"MUJOCO_EGL_DEVICE_ID": mujoco_egl_device_ids[device]},
            )
        finally:
            gc.collect()
        ckpt_to_eval = save_ckpt_dir / "final"

        eval_success_rate = None
        if args.skip_evaluation:
            print(f"[{step_name}] --skip_evaluation set; skipping eval.")
        elif not test_scene_paths:
            print(f"[{step_name}] No test scenes found at {test_scene_dir}; skipping eval.")
        elif not args.evaluate_every_round and target_step % evaluate_dataset_step_number != 0:
            print(
                f"[{step_name}] Skipping eval; {target_step} is not a multiple of "
                f"{evaluate_dataset_step_number}."
            )
        else:
            eval_success_rate, successes, total_evals = run_test_rollout(
                step_name=step_name,
                test_scene_paths=test_scene_paths,
                devices=devices,
                mujoco_egl_device_ids=mujoco_egl_device_ids,
                rollout_prefix=rollout_prefix,
                ckpt_to_eval=ckpt_to_eval,
                policy_cfg_path=policy_cfg_path,
                policy_name=policy_name,
                representation=representation,
                eval_module_name=eval_module,
                task_name=task_name,
                tasks_config_path=tasks_config_path,
                python_executable=python_executable,
                log_video=not args.not_log_video,
                use_relative_action=policy_runtime.use_relative_action,
                rollout_batch_size=args.rollout_batch_size,
                no_ee=args.no_ee,
            )

        if eval_success_rate is not None and not args.disable_early_stopping:
            if total_evals > 0 and successes == total_evals:
                print(
                    f"[{step_name}] Stopping: test rollout reached 100% "
                    f"success ({successes}/{total_evals})."
                )
                return

            if eval_success_rate > args.success_rate_threshold:
                if not eval_every_iter:
                    print(
                        f"[{step_name}] Success rate exceeded "
                        f"{args.success_rate_threshold * 100:.0f}%; "
                        "tracking high-success-regime improvement."
                    )
                eval_every_iter = True

            if eval_every_iter:
                if best_eval_success_rate is None or eval_success_rate > best_eval_success_rate:
                    best_eval_success_rate = eval_success_rate
                    rollouts_since_best = 1
                    print(
                        f"[{step_name}] New best success rate: "
                        f"{best_eval_success_rate * 100:.2f}% "
                        f"({successes}/{total_evals})."
                    )
                else:
                    rollouts_since_best += 1
                    print(
                        f"[{step_name}] No success-rate improvement "
                        f"for {rollouts_since_best} rollout(s); "
                        f"best={best_eval_success_rate * 100:.2f}%."
                    )

                if rollouts_since_best >= NO_IMPROVEMENT_ROLLOUT_PATIENCE:
                    print(
                        f"[{step_name}] Stopping: success rate has not improved "
                        f"for {NO_IMPROVEMENT_ROLLOUT_PATIENCE} rollouts in the "
                        f">{args.success_rate_threshold * 100:.0f}% regime."
                    )
                    return


if __name__ == "__main__":
    main()

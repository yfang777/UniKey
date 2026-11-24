import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from policy.policy_training.action_ensemble_controller import (
    ActionEnsembleController,
)
from policy.policy_training.env_sim import SimEnv
from keypoint_program_solver.utils import concatenate_camera_frames


from policy.utils.video import SuppressNativeStderr, write_rollout_frame


def _rot6d_to_rotmat_np(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def abs_action_10d_to_13d_np(action_10d):
    action = np.asarray(action_10d, dtype=np.float32)
    out = np.zeros(13)
    out[:3] = action[:3]
    out[3:12] = _rot6d_to_rotmat_np(action[3:9]).reshape(9)
    out[12] = float(action[9])
    return out


def build_row_arrays(step_idx, xml_path, lerobot_feature):
    row = {
        "meta.step": int(step_idx),
        "meta.xml_path": str(xml_path),
    }
    for key, value in lerobot_feature.items():
        dtype = np.float64 if key == "state_buf" else np.float32
        row[key] = np.asarray(value, dtype=dtype).reshape(-1)
    return row


def close_rollout_videos(container, stream):
    with SuppressNativeStderr():
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
        if container is not None:
            container.close()


def write_rollout_parquet(rollout):
    import pandas as pd

    pd.DataFrame(rollout["lerobot_feature_rows"]).to_parquet(
        rollout["parquet_path"],
        index=False,
    )


def make_rollout_state(
    rollout_spec,
    setting_cfg,
    eval_cfg,
    camera_names,
    action_ensemble,
    max_ee_linear_vel,
    dummy_simulate_t=0.0,
):
    chunk_size = int(setting_cfg["chunk_size"])
    policy_infer_freq = int(setting_cfg["policy_infer_freq"])
    control_freq = int(setting_cfg["control_freq"])
    action_timestamp_offset = float(setting_cfg["action_timestamp_offset"])
    action_decay_coeff = setting_cfg["action_decay_coeff"]

    env = SimEnv(max_ee_linear_vel=max_ee_linear_vel)
    obs, reset_info = env.reset(
        mujoco_xml_path=rollout_spec["xml_path"],
        log_video=True,
        camera_names=camera_names,
        t_horizon=eval_cfg["t_horizon"],
        dummy_simulate_t=dummy_simulate_t,
    )

    controller = ActionEnsembleController(
        policy_chunk_size=chunk_size,
        policy_fps=policy_infer_freq,
        env_fps=control_freq,
        action_dim=10,
        coeff=action_decay_coeff,
        action_timestamp_offset=action_timestamp_offset,
        initial_action=reset_info["initial_action"],
        ensemble=action_ensemble,
    )

    return {
        "xml_path": rollout_spec["xml_path"],
        "parquet_path": Path(rollout_spec["parquet_path"]),
        "video_path": Path(rollout_spec["video_path"]),
        "env": env,
        "obs": obs,
        "done": False,
        "truncated": False,
        "step": 0,
        "lerobot_feature_rows": [],
        "controller": controller,
        "container": None,
        "stream": None,
    }


def run_rgb_rollout_worker(
    conn,
    rollout_spec,
    setting_cfg,
    eval_cfg,
    log_video,
    camera_names,
    action_ensemble,
    max_ee_linear_vel,
    write_parquet=True,
    dummy_simulate_t=0.0,
):
    policy_infer_freq = int(setting_cfg["policy_infer_freq"])
    control_freq = int(setting_cfg["control_freq"])
    assert control_freq % policy_infer_freq == 0

    rollout = None
    env_step_time = 0.0
    worker_start_time = time.perf_counter()

    try:
        rollout = make_rollout_state(
            rollout_spec,
            setting_cfg,
            eval_cfg,
            camera_names,
            action_ensemble,
            max_ee_linear_vel=max_ee_linear_vel,
            dummy_simulate_t=dummy_simulate_t,
        )
        conn.send({"obs": rollout["obs"]})

        while True:
            message = conn.recv()
            if message["type"] == "close":
                break

            chunk_np = message["action_chunk"]
            actions_to_execute = rollout["controller"].step(chunk_np)

            num_actions = len(actions_to_execute)
            for idx, action_10d in enumerate(actions_to_execute):
                is_render_step = idx == num_actions - 1
                abs_action_13d = abs_action_10d_to_13d_np(action_10d)

                step_start_time = time.perf_counter()
                obs, _, done, truncated, _ = rollout["env"].step(
                    abs_action_13d,
                    no_render=not is_render_step,
                )
                env_step_time += time.perf_counter() - step_start_time

                rollout["obs"] = obs
                rollout["done"] = done
                rollout["truncated"] = truncated
                rollout["step"] += 1
                rollout["lerobot_feature_rows"].append(
                    build_row_arrays(
                        step_idx=rollout["step"],
                        xml_path=rollout["xml_path"],
                        lerobot_feature=rollout["env"].lerobot_features[-1],
                    )
                )

                if is_render_step and log_video:
                    frames = [
                        obs[f"observation.images.{camera_name}"]
                        for camera_name in camera_names
                    ]
                    combined_frame = concatenate_camera_frames(frames)
                    container, stream = write_rollout_frame(
                        rollout["container"],
                        rollout["stream"],
                        rollout["video_path"],
                        combined_frame,
                        policy_infer_freq,
                    )
                    rollout["container"] = container
                    rollout["stream"] = stream

                if done or truncated:
                    break

            if rollout["done"] or rollout["truncated"]:
                if log_video:
                    close_rollout_videos(
                        rollout["container"],
                        rollout["stream"],
                    )
                    rollout["container"] = None
                    rollout["stream"] = None
                if write_parquet:
                    write_rollout_parquet(rollout)
                conn.send(
                    {
                        "done": True,
                        "truncated": rollout["truncated"],
                        "step": rollout["step"],
                        "rows_written": len(rollout["lerobot_feature_rows"]),
                        "env_step_time_s": env_step_time,
                        "worker_wall_time_s": time.perf_counter() - worker_start_time,
                        "xml_path": str(rollout["xml_path"]),
                        "parquet_path": str(rollout["parquet_path"]),
                        "video_path": str(rollout["video_path"]),
                    }
                )
                break

            conn.send({"done": False, "obs": rollout["obs"]})
    except Exception as exc:
        conn.send({"error": repr(exc)})
        raise
    finally:
        if rollout is not None:
            if log_video and rollout["container"] is not None:
                close_rollout_videos(
                    rollout["container"],
                    rollout["stream"],
                )
            rollout["env"].close()
        conn.close()

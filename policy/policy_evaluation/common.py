import multiprocessing as mp
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import torch

from policy.policy_training.env_sim import SimEnv
from policy.policy_training.action_ensemble_controller import (
    ActionEnsembleController,
)
from policy.policy_evaluation.rollout_worker import (
    build_row_arrays, run_rgb_rollout_worker as run_rgb_rollout_worker_lean,
)
from policy.policy_training.preprocessing import resize_rgb_tensor
from policy.policy_training.utils import rel_to_abs_actions, rot6d_to_rotmat
from keypoint_program_solver.utils import concatenate_camera_frames


from policy.utils.video import SuppressNativeStderr, add_dense_video_stream, write_rollout_frame


def named_scenes(paths):
    """Keep dataset scene names and disambiguate scenes drawn from multiple splits."""
    names = [Path(path).parent.name for path in paths]
    if len(set(names)) < len(names):
        names = [f"{index}_{name}" for index, name in enumerate(names)]
    return list(zip(paths, names))


def image_tensor_from_obs(obs, camera_name, device, resize_rgb_image=False):
    bgr = obs[f"observation.images.{camera_name}"]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        / 255.0
    )
    if resize_rgb_image:
        return resize_rgb_tensor(image)
    return image


def image_tensor_batch_from_obs_list(
    obs_list,
    camera_name,
    device,
    resize_rgb_image=False,
):
    rgbs = [
        cv2.cvtColor(obs[f"observation.images.{camera_name}"], cv2.COLOR_BGR2RGB)
        for obs in obs_list
    ]
    image = (
        torch.from_numpy(np.ascontiguousarray(np.stack(rgbs)))
        .float()
        .permute(0, 3, 1, 2)
        .to(device)
        / 255.0
    )
    if resize_rgb_image:
        return resize_rgb_tensor(image)
    return image


def policy_observation(obs, camera_names, device, no_ee, resize_rgb_image=False):
    state = torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to(device)
    policy_state = state
    if no_ee:
        policy_state = torch.zeros_like(state)

    observation = {"observation.state": policy_state}
    for camera_name in camera_names:
        observation[f"observation.images.{camera_name}"] = image_tensor_from_obs(
            obs,
            camera_name,
            device,
            resize_rgb_image=resize_rgb_image,
        )
    return observation, state


def policy_observation_batch(
    obs_list,
    camera_names,
    device,
    no_ee,
    resize_rgb_image=False,
):
    state = torch.from_numpy(
        np.stack([obs["observation.state"] for obs in obs_list])
    ).float().to(device)
    policy_state = state
    if no_ee:
        policy_state = torch.zeros_like(state)

    observation = {"observation.state": policy_state}
    for camera_name in camera_names:
        observation[f"observation.images.{camera_name}"] = image_tensor_batch_from_obs_list(
            obs_list,
            camera_name,
            device,
            resize_rgb_image=resize_rgb_image,
        )
    return observation, state


def abs_action_10d_to_13d(action_10d):
    action = torch.from_numpy(np.asarray(action_10d)).float()
    pos = action[0:3].numpy()
    rot_9d = rot6d_to_rotmat(action[3:9]).numpy().reshape(9)
    gripper = float(action[9])
    out = np.zeros(13)
    out[:3] = pos
    out[3:12] = rot_9d
    out[12] = gripper
    return out


def write_rollout_camera_frames(containers, streams, camera_video_paths, frames,
                                policy_infer_freq):
    with SuppressNativeStderr():
        for camera_name, frame in frames.items():
            if camera_name not in containers:
                video_path = Path(camera_video_paths[camera_name])
                video_path.parent.mkdir(parents=True, exist_ok=True)
                container = av.open(str(video_path), mode="w")
                h, w = frame.shape[:2]
                containers[camera_name] = container
                streams[camera_name] = add_dense_video_stream(
                    container,
                    w,
                    h,
                    policy_infer_freq,
                )

            av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in streams[camera_name].encode(av_frame):
                containers[camera_name].mux(packet)


def close_rollout_videos(container, stream, camera_containers, camera_streams):
    with SuppressNativeStderr():
        for camera_name, camera_container in camera_containers.items():
            camera_stream = camera_streams[camera_name]
            for packet in camera_stream.encode():
                camera_container.mux(packet)
            camera_container.close()
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
        if container is not None:
            container.close()


def run_rgb_inference(
    policy,
    parquet_path,
    video_path,
    xml_path,
    setting_cfg,
    eval_cfg,
    device,
    log_video,
    camera_names,
    no_ee,
    use_absolute_action,
    action_ensemble,
    max_ee_linear_vel,
    camera_video_paths=None,
    log_video_every_step=False,
    resize_rgb_image=False,
    dummy_simulate_t=0.0,
):
    chunk_size = int(setting_cfg["chunk_size"])
    policy_infer_freq = int(setting_cfg["policy_infer_freq"])
    control_freq = int(setting_cfg["control_freq"])
    action_timestamp_offset = float(setting_cfg["action_timestamp_offset"])
    t_horizon = eval_cfg["t_horizon"]
    action_decay_coeff = setting_cfg["action_decay_coeff"]
    video_fps = control_freq if log_video_every_step else policy_infer_freq
    assert control_freq % policy_infer_freq == 0

    env = SimEnv(max_ee_linear_vel=max_ee_linear_vel)
    policy.reset()

    obs, reset_info = env.reset(
        mujoco_xml_path=xml_path,
        log_video=True,
        camera_names=camera_names,
        t_horizon=t_horizon,
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

    container = None
    stream = None
    camera_containers = {}
    camera_streams = {}
    step = 0
    done, truncated = False, False
    lerobot_feature_rows = []

    while not (done or truncated):
        observation, state = policy_observation(
            obs,
            camera_names,
            device,
            no_ee,
            resize_rgb_image=resize_rgb_image,
        )

        with torch.inference_mode():
            action = policy.predict_action_chunk(observation)

        if use_absolute_action:
            action_abs = action
        else:
            action_abs = rel_to_abs_actions(
                {
                    "observation.state": state.unsqueeze(1),
                    "action": action,
                }
            )["action"]

        chunk_np = action_abs[0].cpu().numpy()
        actions_to_execute = controller.step(chunk_np)

        num_actions = len(actions_to_execute)
        for idx, action_10d in enumerate(actions_to_execute):
            is_render_step = log_video_every_step or idx == num_actions - 1
            abs_action_13d = abs_action_10d_to_13d(action_10d)
            obs, _, done, truncated, _ = env.step(
                abs_action_13d,
                no_render=not is_render_step,
            )
            step += 1

            lerobot_feature_rows.append(
                build_row_arrays(
                    step_idx=step,
                    xml_path=xml_path,
                    lerobot_feature=env.lerobot_features[-1],
                )
            )

            if is_render_step and log_video:
                frames = {
                    camera_name: obs[f"observation.images.{camera_name}"]
                    for camera_name in camera_names
                }
                if camera_video_paths is not None:
                    write_rollout_camera_frames(
                        camera_containers,
                        camera_streams,
                        camera_video_paths,
                        frames,
                        video_fps,
                    )
                if video_path is not None:
                    combined_frame = concatenate_camera_frames(list(frames.values()))
                    container, stream = write_rollout_frame(
                        container,
                        stream,
                        video_path,
                        combined_frame,
                        video_fps,
                    )

            if done or truncated:
                break

    if log_video:
        close_rollout_videos(
            container,
            stream,
            camera_containers,
            camera_streams,
        )

    pd.DataFrame(lerobot_feature_rows).to_parquet(parquet_path, index=False)
    env.close()


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
        "parquet_path": rollout_spec["parquet_path"],
        "video_path": rollout_spec["video_path"],
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


def run_rgb_inference_batch(
    policy,
    rollout_specs,
    setting_cfg,
    eval_cfg,
    device,
    log_video,
    camera_names,
    no_ee,
    use_absolute_action,
    action_ensemble,
    max_ee_linear_vel,
    resize_rgb_image=False,
    dummy_simulate_t=0.0,
):
    policy_infer_freq = int(setting_cfg["policy_infer_freq"])
    control_freq = int(setting_cfg["control_freq"])
    assert control_freq % policy_infer_freq == 0

    policy.reset()
    rollouts = [
        make_rollout_state(
            rollout_spec,
            setting_cfg,
            eval_cfg,
            camera_names,
            action_ensemble,
            max_ee_linear_vel=max_ee_linear_vel,
            dummy_simulate_t=dummy_simulate_t,
        )
        for rollout_spec in rollout_specs
    ]

    while any(not (rollout["done"] or rollout["truncated"]) for rollout in rollouts):
        active_indices = [
            idx
            for idx, rollout in enumerate(rollouts)
            if not (rollout["done"] or rollout["truncated"])
        ]
        active_obs = [rollouts[idx]["obs"] for idx in active_indices]
        observation, state = policy_observation_batch(
            active_obs,
            camera_names,
            device,
            no_ee,
            resize_rgb_image=resize_rgb_image,
        )

        with torch.inference_mode():
            action = policy.predict_action_chunk(observation)

        if use_absolute_action:
            action_abs = action
        else:
            action_abs = rel_to_abs_actions(
                {
                    "observation.state": state.unsqueeze(1),
                    "action": action,
                }
            )["action"]

        action_chunks = action_abs.cpu().numpy()
        for batch_idx, rollout_idx in enumerate(active_indices):
            rollout = rollouts[rollout_idx]
            chunk_np = action_chunks[batch_idx]
            actions_to_execute = rollout["controller"].step(chunk_np)

            num_actions = len(actions_to_execute)
            for idx, action_10d in enumerate(actions_to_execute):
                is_render_step = idx == num_actions - 1
                abs_action_13d = abs_action_10d_to_13d(action_10d)
                obs, _, done, truncated, _ = rollout["env"].step(
                    abs_action_13d,
                    no_render=not is_render_step,
                )
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

    for rollout in rollouts:
        if log_video:
            close_rollout_videos(
                rollout["container"],
                rollout["stream"],
                {},
                {},
            )
        pd.DataFrame(rollout["lerobot_feature_rows"]).to_parquet(
            rollout["parquet_path"],
            index=False,
        )
        rollout["env"].close()


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
                abs_action_13d = abs_action_10d_to_13d(action_10d)

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
                        {},
                        {},
                    )
                    rollout["container"] = None
                    rollout["stream"] = None
                if write_parquet:
                    pd.DataFrame(rollout["lerobot_feature_rows"]).to_parquet(
                        rollout["parquet_path"],
                        index=False,
                    )
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
                    {},
                    {},
                )
            rollout["env"].close()
        conn.close()


def run_rgb_inference_batch_multiprocess(
    policy,
    rollout_specs,
    setting_cfg,
    eval_cfg,
    device,
    log_video,
    camera_names,
    no_ee,
    use_absolute_action,
    action_ensemble,
    max_ee_linear_vel,
    resize_rgb_image=False,
    write_parquet=True,
    dummy_simulate_t=0.0,
):
    policy.reset()
    ctx = mp.get_context("forkserver")
    workers = []

    for rollout_spec in rollout_specs:
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=run_rgb_rollout_worker_lean,
            args=(
                child_conn,
                rollout_spec,
                setting_cfg,
                eval_cfg,
                log_video,
                camera_names,
                action_ensemble,
                max_ee_linear_vel,
                write_parquet,
                dummy_simulate_t,
            ),
        )
        process.start()
        child_conn.close()
        workers.append(
            {
                "conn": parent_conn,
                "process": process,
                "obs": None,
                "done": False,
                "result": None,
            }
        )

    completed = False
    try:
        for worker in workers:
            message = worker["conn"].recv()
            if "error" in message:
                raise RuntimeError(message["error"])
            worker["obs"] = message["obs"]

        while any(not worker["done"] for worker in workers):
            active_indices = [
                idx for idx, worker in enumerate(workers) if not worker["done"]
            ]
            active_obs = [workers[idx]["obs"] for idx in active_indices]
            observation, state = policy_observation_batch(
                active_obs,
                camera_names,
                device,
                no_ee,
                resize_rgb_image=resize_rgb_image,
            )

            with torch.inference_mode():
                action = policy.predict_action_chunk(observation)

            if use_absolute_action:
                action_abs = action
            else:
                action_abs = rel_to_abs_actions(
                    {
                        "observation.state": state.unsqueeze(1),
                        "action": action,
                    }
                )["action"]

            action_chunks = action_abs.cpu().numpy()
            for batch_idx, worker_idx in enumerate(active_indices):
                workers[worker_idx]["conn"].send(
                    {
                        "type": "action_chunk",
                        "action_chunk": action_chunks[batch_idx],
                    }
                )

            for worker_idx in active_indices:
                message = workers[worker_idx]["conn"].recv()
                if "error" in message:
                    raise RuntimeError(message["error"])
                if message["done"]:
                    workers[worker_idx]["done"] = True
                    workers[worker_idx]["result"] = message
                else:
                    workers[worker_idx]["obs"] = message["obs"]

        for worker in workers:
            worker["process"].join()
            if worker["process"].exitcode != 0:
                raise RuntimeError(
                    f"rollout worker exited with code {worker['process'].exitcode}"
                )
        completed = True
        return [worker["result"] for worker in workers]
    finally:
        if not completed:
            for worker in workers:
                if worker["process"].is_alive():
                    worker["process"].terminate()
                worker["process"].join()
        for worker in workers:
            worker["conn"].close()

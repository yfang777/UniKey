"""Divergence detection and dataset/video helpers for corrective collection."""

from policy.utils.video import SuppressNativeStderr, add_dense_video_stream
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd


def is_diverge(
    policy_keypoints,
    planner_keypoints,
    pos_diff_threshold,
    policy2planner_factor,
):
    """Detect divergence between policy and planner end-effector keypoint
    trajectories.

    The divergence metric is the average L2 position error across three end-effector keypoints (indices
    [0, -2, -1] of the per-frame keypoints array — corresponding to the
    tracked ee site and the gripper left/right tips). Rotation is not used,
    so there is no `rot_diff_threshold` argument.

    `policy_keypoints` and `planner_keypoints` are per-frame
    `observation.keypoints` arrays; each may come in as a flattened (T, N*3)
    vector (as stored in the lerobot parquet) or already as (T, N, 3).

    Steps are measured in planner-fps; the policy is interpolated to
    planner-fps via `policy2planner_factor` (planner_fps / policy_fps).
    Returns (diverged_bool, planner_first_diverge_idx, policy_first_diverge_idx).
    """
    num_planner_steps = len(planner_keypoints)
    num_policy_steps = len(policy_keypoints)

    policy_kp = np.asarray(policy_keypoints)
    planner_kp = np.asarray(planner_keypoints)
    if policy_kp.ndim == 2:
        policy_kp = policy_kp.reshape(num_policy_steps, -1, 3)
    if planner_kp.ndim == 2:
        planner_kp = planner_kp.reshape(num_planner_steps, -1, 3)

    ee_indices = [0, -2, -1]
    policy_ee = policy_kp[:, ee_indices, :]
    planner_ee = planner_kp[:, ee_indices, :]

    eval_limit = num_planner_steps - 1
    eval_indices = np.arange(int(eval_limit) + 1)
    eval_policy_query_indices = eval_indices / policy2planner_factor

    interpolated_policy_ee = np.zeros((len(eval_indices), 3, 3))
    for k in range(3):
        for dim in range(3):
            interpolated_policy_ee[:, k, dim] = np.interp(
                eval_policy_query_indices,
                np.arange(num_policy_steps),
                policy_ee[:, k, dim],
            )

    diff = interpolated_policy_ee - planner_ee
    l2_per_kp = np.linalg.norm(diff, axis=2)
    avg_l2 = np.mean(l2_per_kp, axis=1)

    mismatch_indices = np.where(avg_l2 > pos_diff_threshold)[0]
    if len(mismatch_indices) > 0:
        planner_first_diverge_idx = mismatch_indices[0]
        policy_first_diverge_idx = int(planner_first_diverge_idx / policy2planner_factor)
        return True, planner_first_diverge_idx, policy_first_diverge_idx

    return False, -1, -1


def stage_containing_frame(stages, planner_frame: int):
    """Return the 1-indexed stage containing `planner_frame`, or None."""
    for s in stages:
        if s["start"] <= planner_frame < max(s["end"], s["start"] + 1):
            return s["stage"]
    if stages and planner_frame >= stages[-1]["end"]:
        return stages[-1]["stage"]
    return None


def planner_features_to_dataframe(features):
    """Convert a list of lerobot_feature dicts (with image keys) to a DataFrame
    suitable for parquet storage (images and pointcloud columns dropped)."""
    df = pd.DataFrame(features)
    drop_cols = [c for c in df.columns
                 if c == "render_qpos" or c.startswith("observation.images.")
                 or c.startswith("observation.pointcloud")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def dense_camera_video_paths(output_dir, cam_names):
    return {
        f"observation.images.{cam}": str(
            Path(output_dir)
            / "videos"
            / "chunk-000"
            / f"observation.images.{cam}"
            / "episode_000000.mp4"
        )
        for cam in cam_names
    }


def encode_features_camera_videos(features, output_dir, cam_names, fps: int):
    """Encode planner lerobot_features as per-camera AV1 videos."""
    camera_videos = dense_camera_video_paths(output_dir, cam_names)
    if not cam_names:
        return camera_videos

    containers = {}
    streams = {}
    with SuppressNativeStderr():
        try:
            for feat in features:
                for cam in cam_names:
                    key = f"observation.images.{cam}"
                    if key not in feat or feat[key] is None:
                        continue
                    rgb = np.asarray(feat[key])
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.shape[-1] == 3 else rgb
                    if key not in containers:
                        video_path = Path(camera_videos[key])
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        container = av.open(str(video_path), mode="w")
                        h, w = bgr.shape[:2]
                        containers[key] = container
                        streams[key] = add_dense_video_stream(container, w, h, fps)
                    av_frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
                    for packet in streams[key].encode(av_frame):
                        containers[key].mux(packet)
        finally:
            for key, container in containers.items():
                stream = streams[key]
                for packet in stream.encode():
                    container.mux(packet)
                container.close()

    return camera_videos

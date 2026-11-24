"""corrective recovery collection using recorded stable motions for both recovery passes.

Restore the last successful policy state and execute the remaining saved stages
for the reference trajectory. Detect divergence with the end-effector keypoint metric,
then restore the divergence state and resume the same stable sequence there.
Preserve stage judging, frame accounting, time budgets, and dataset case codes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import runpy
import time
from contextlib import ExitStack

import mujoco
import yaml

from keypoint_program_solver.env import ReKepEnv
from keypoint_program_solver.solver import Solver, body_world_pose
from keypoint_program_solver.utils import CameraRenderers
from policy.evaluate import bind_helpers, evaluate_episode_augment, load_module

from .utils import (
    encode_features_camera_videos,
    is_diverge,
    planner_features_to_dataframe,
    stage_containing_frame,
)


# Dataset case codes are stable across collection runs.
CASE_OK = 0
CASE_NO_DIVERGE = 1
CASE_TOO_SHORT = 2
CASE_WORKER_CRASH = 3
CASE_PLANNER_D_FAIL = 4
CASE_RECOVERY_FAIL = 5
CASE_OK_PARTIAL = 6
CASE_DESCRIPTIONS = {
    CASE_OK: "recovery data generated",
    CASE_NO_DIVERGE: "policy failed but is_diverge returned False",
    CASE_TOO_SHORT: "recovery trajectory shorter than required seconds",
    CASE_WORKER_CRASH: "worker crashed",
    CASE_PLANNER_D_FAIL: (
        "step-(d) stable reference did not pass constraints_judge or was "
        "early-stopped by t_horizon budget"
    ),
    CASE_RECOVERY_FAIL: (
        "step-(e) stable recovery did not pass constraints_judge or was "
        "early-stopped by t_horizon budget"
    ),
    CASE_OK_PARTIAL: (
        "recovery generated below target length but >= "
        "prediction_horizon_frames (\"0_2\" — partial but trainable)"
    ),
}


def _write_recovery_info(scene_output_dir: Path, info: dict) -> None:
    path = Path(scene_output_dir) / "recovery_info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def write_policy_pass_info(scene_output_dir, xml_path, policy_parquet_path,
                            representation, recovery_seconds, planner_fps):
    """Mark scenes whose policy passed (no recovery needed)."""
    info = {
        "xml_path": str(xml_path),
        "policy_parquet_path": str(policy_parquet_path),
        "representation": representation,
        "recovery_seconds": float(recovery_seconds),
        "planner_fps": int(planner_fps),
        "required_recovery_frames": int(round(recovery_seconds * planner_fps)),
        "case": CASE_OK,
        "case_description": "policy passed (no recovery needed)",
        "policy_passed": True,
        "diverged": False,
        "planner_diverge_idx": -1,
        "policy_diverge_idx": -1,
        "stage_containing_diverge": -1,
        "recovery_length_frames": 0,
        "recovery_parquet_path": None,
    }
    _write_recovery_info(Path(scene_output_dir), info)
    return info


class StepBudgetExceeded(Exception):
    pass


class RecoveryEnv(ReKepEnv):
    viewer = None

    def step(self, action, n=1):
        remaining = self.step_budget - round(self.data.time / self.dt)
        steps = min(n, remaining)
        start = time.monotonic()
        super().step(action, steps)
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(max(0, steps * self.dt / self.realtime_factor - (time.monotonic() - start)))
        if steps < n:
            raise StepBudgetExceeded()


def rendered_frames(env, renderers):
    for frame in env.lerobot_features:
        env.data.qpos[:] = frame["render_qpos"]
        mujoco.mj_kinematics(env.model, env.data)
        mujoco.mj_camlight(env.model, env.data)
        images = {}
        for camera in env.cams:
            renderer = renderers.for_camera(camera)
            renderer.update_scene(env.data, camera=camera)
            images[f"observation.images.{camera}"] = renderer.render()
        yield images


def run_stable_recovery(xml_path, stable_solution_path, output_dir, planner_fps,
                       rekep_config_path, file_id, start_stage,
                       start_state_buf, start_gripper_value, t_horizon,
                       constraint_judge_path, helper_func_path,
                       rekep_tolerance, max_ee_linear_vel,
                       render_realtime=False, realtime_factor=1.0, start_context=None):
    """Restore an anchor, replay the remaining stages and export the judged trajectory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{file_id}.parquet"
    stages_path = output_dir / "stages.json"
    generation_path = output_dir / "generation.json"

    config = yaml.safe_load(Path(rekep_config_path).read_text())
    solver = Solver({}, config)
    env = RecoveryEnv(xml_path, max_ee_linear_vel, config, 0.0)
    initial = {name: body_world_pose(env, name) for name in env.object_map}
    sequence = runpy.run_path(str(stable_solution_path))["STAGES"]
    if not 1 <= start_stage <= len(sequence) + 1 or t_horizon <= 0:
        raise ValueError("Invalid recovery stage or horizon")
    state = np.array(start_state_buf, dtype=np.float64, copy=True)
    mujoco.mj_setState(env.model, env.data, state, env._full_state_mask())
    env.data.time = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.render_qpos[:] = env.data.qpos
    env.last_gripper_status = float(np.asarray(start_gripper_value).item())
    if start_context is not None:
        env.episode_context = np.array(start_context, copy=True)
    env.step_budget = round(t_horizon / env.dt)
    stages = []
    early_stopped = False
    with ExitStack() as stack:
        if render_realtime:
            from mujoco import viewer
            if realtime_factor <= 0:
                raise ValueError("realtime_factor must be positive")
            env.viewer = stack.enter_context(viewer.launch_passive(env.model, env.data))
            env.realtime_factor = realtime_factor
        for stage in sequence[start_stage - 1:]:
            start = len(env.lerobot_features)
            try:
                solver.run_stages(env, [stage], initial=initial)
            except StepBudgetExceeded:
                early_stopped = True
            kind = {"open": "release", "close": "grasp"}.get(stage.get("action"), "motion")
            stages.append(dict(stage=stage["stage"], type=kind, start=start, end=len(env.lerobot_features)))
            if early_stopped:
                break
    env.lerobot_features.append(env.get_lerobot_feature())
    judge = load_module(constraint_judge_path, "stable_recovery_judge")
    helpers = load_module(helper_func_path, "stable_recovery_helpers")
    bind_helpers(judge, helpers)
    df = planner_features_to_dataframe(env.lerobot_features)
    passed, last_stage, last_frame = evaluate_episode_augment(
        df, judge,
        judge.grasp_stage, float(rekep_tolerance), int(start_stage) - 1,
    )
    if early_stopped:
        reason = "stable execution exceeded the simulation step budget"
    elif not passed:
        reason = "stable execution did not pass constraints_judge_augment"
    else:
        reason = None
    generation_info = {
        "mode": "stable_augment", "stable_solution_path": str(stable_solution_path),
        "anchor_stage": int(start_stage), "t_horizon": float(t_horizon),
        "step_budget": int(round(float(t_horizon) / env.dt)),
        "step_counter": round(env.data.time / env.dt),
        "early_stopped": bool(early_stopped), "rekep_tolerance": float(rekep_tolerance),
        "passed": bool(passed), "last_success_stage": int(last_stage),
        "last_success_frame": int(last_frame), "reason": reason,
    }

    cam_names = list(env.cams)
    df.to_parquet(parquet_path, index=False)
    renderers = CameraRenderers(env.model)
    try:
        camera_videos = encode_features_camera_videos(rendered_frames(env, renderers), output_dir, cam_names, planner_fps)
    finally:
        renderers.close()

    with open(stages_path, "w") as f:
        json.dump({
            "xml_path": str(xml_path),
            "start_stage": int(start_stage),
            "stages": stages,
            "early_stopped": bool(early_stopped),
        }, f, indent=2)
    with open(generation_path, "w") as f:
        json.dump(generation_info, f, indent=2)

    return len(df), parquet_path, camera_videos, stages, generation_info


def generate_corrective_recovery(
    xml_path,
    policy_parquet_path,
    scene_output_dir,
    representation,
    stable_solution_path,
    constraint_judge_path,
    helper_func_path,
    rekep_config_path,
    pos_diff_threshold,
    policy2planner_factor,
    recovery_seconds,
    planner_fps,
    prediction_horizon_frames,
    t_horizon,
    rekep_tolerance,
    max_ee_linear_vel,
    file_id,
    last_success_stage,
    last_success_frame,
    render_realtime=False,
    realtime_factor=1.0,
):
    """Generate a stable reference and stable recovery using keypoint divergence."""
    scene_output_dir = Path(scene_output_dir)
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    planner_d_dir = scene_output_dir / "planner_d"
    recovery_dir = scene_output_dir / "recovery"

    info = {
        "xml_path": str(xml_path),
        "policy_parquet_path": str(policy_parquet_path),
        "representation": representation,
        "stable_solution_path": str(stable_solution_path),
        "recovery_seconds": float(recovery_seconds),
        "planner_fps": int(planner_fps),
        "t_horizon": float(t_horizon),
        "rekep_tolerance": float(rekep_tolerance),
        "target_recovery_frames": int(recovery_seconds * planner_fps),
        "prediction_horizon_frames": int(prediction_horizon_frames),
        "last_success_stage": int(last_success_stage),
        "last_success_frame": int(last_success_frame),
        "case": None,
        "case_description": None,
        "diverged": False,
        "planner_d_diverge_idx": -1,
        "policy_diverge_idx": -1,
        "stage_d_at_diverge": -1,
        "recovery_length_frames": 0,
        "planner_d_parquet_path": None,
        "planner_d_camera_videos": None,
        "recovery_parquet_path": None,
        "recovery_camera_videos": None,
        "planner_d_generation": None,
        "recovery_generation": None,
    }

    policy_df = pd.read_parquet(policy_parquet_path)
    if last_success_frame < 0 or last_success_frame >= len(policy_df):
        raise ValueError(
            f"last_success_frame={last_success_frame} out of range for "
            f"policy rollout of length {len(policy_df)}"
        )
    if "observation.keypoints" not in policy_df.columns:
        raise KeyError(
            f"policy parquet {policy_parquet_path} is missing "
            "'observation.keypoints' — corrective divergence detection requires it."
        )

    start_state_buf_d = np.asarray(
        policy_df.iloc[last_success_frame]["state_buf"], dtype=np.float64
    )
    start_gripper_value_d = np.asarray(
        policy_df.iloc[last_success_frame]["last_gripper_status"], dtype=np.float64
    )

    _, planner_d_parquet, planner_d_camera_videos, stages_d, planner_d_gen = (
        run_stable_recovery(
            xml_path=xml_path,
            stable_solution_path=stable_solution_path,
            output_dir=planner_d_dir,
            planner_fps=planner_fps,
            rekep_config_path=rekep_config_path,
            file_id=file_id,
            start_stage=int(last_success_stage) + 1,
            start_state_buf=start_state_buf_d,
            start_context=policy_df.iloc[last_success_frame].get("observation.constraint_context"),
            start_gripper_value=start_gripper_value_d,
            t_horizon=float(t_horizon),
            constraint_judge_path=constraint_judge_path,
            helper_func_path=helper_func_path,
            rekep_tolerance=float(rekep_tolerance),
            max_ee_linear_vel=max_ee_linear_vel,
            render_realtime=bool(render_realtime),
            realtime_factor=float(realtime_factor),
        )
    )
    info["planner_d_parquet_path"] = str(planner_d_parquet)
    info["planner_d_camera_videos"] = planner_d_camera_videos
    info["planner_d_generation"] = planner_d_gen

    planner_d_usable = (
        bool(planner_d_gen["passed"]) and not bool(planner_d_gen["early_stopped"])
    )
    if not planner_d_usable:
        info["case"] = CASE_PLANNER_D_FAIL
        info["case_description"] = (
            f"{CASE_DESCRIPTIONS[CASE_PLANNER_D_FAIL]}: "
            f"{planner_d_gen['reason']}"
        )
        _write_recovery_info(scene_output_dir, info)
        return info

    planner_d_df = pd.read_parquet(planner_d_parquet)
    if "observation.keypoints" not in planner_d_df.columns:
        raise KeyError(
            f"planner_d parquet {planner_d_parquet} is missing "
            "'observation.keypoints' — corrective divergence detection requires it."
        )
    planner_d_keypoints = np.stack(planner_d_df["observation.keypoints"].values)
    policy_keypoints_full = np.stack(policy_df["observation.keypoints"].values)
    policy_keypoints_after = policy_keypoints_full[last_success_frame:]

    diverged, planner_d_diverge_idx, policy_diverge_offset = is_diverge(
        policy_keypoints_after, planner_d_keypoints,
        pos_diff_threshold, policy2planner_factor,
    )
    policy_diverge_idx_abs = (
        int(policy_diverge_offset) + int(last_success_frame) if diverged else -1
    )
    info.update({
        "diverged": bool(diverged),
        "planner_d_diverge_idx": int(planner_d_diverge_idx),
        "policy_diverge_idx": int(policy_diverge_idx_abs),
    })

    if not diverged:
        info["case"] = CASE_NO_DIVERGE
        info["case_description"] = CASE_DESCRIPTIONS[CASE_NO_DIVERGE]
        _write_recovery_info(scene_output_dir, info)
        return info

    stage_d_at_diverge = stage_containing_frame(stages_d, int(planner_d_diverge_idx))
    if stage_d_at_diverge is None:
        info["case"] = CASE_NO_DIVERGE
        info["case_description"] = (
            "step-(d) divergence frame is outside the replanned stage range"
        )
        _write_recovery_info(scene_output_dir, info)
        return info
    info["stage_d_at_diverge"] = int(stage_d_at_diverge)

    if policy_diverge_idx_abs >= len(policy_df):
        raise ValueError(
            f"policy_diverge_idx_abs={policy_diverge_idx_abs} >= "
            f"policy rollout length {len(policy_df)}"
        )

    start_state_buf_e = np.asarray(
        policy_df.iloc[policy_diverge_idx_abs]["state_buf"], dtype=np.float64
    )
    start_gripper_value_e = np.asarray(
        policy_df.iloc[policy_diverge_idx_abs]["last_gripper_status"],
        dtype=np.float64,
    )
    recovery_length, recovery_parquet, recovery_camera_videos, _, recovery_gen = run_stable_recovery(
        xml_path=xml_path,
        stable_solution_path=stable_solution_path,
        output_dir=recovery_dir,
        planner_fps=planner_fps,
        rekep_config_path=rekep_config_path,
        file_id=file_id,
        start_stage=int(stage_d_at_diverge),
        start_state_buf=start_state_buf_e,
        start_context=policy_df.iloc[policy_diverge_idx_abs].get("observation.constraint_context"),
        start_gripper_value=start_gripper_value_e,
        t_horizon=float(t_horizon),
        constraint_judge_path=constraint_judge_path,
        helper_func_path=helper_func_path,
        rekep_tolerance=float(rekep_tolerance),
        max_ee_linear_vel=max_ee_linear_vel,
        render_realtime=bool(render_realtime),
        realtime_factor=float(realtime_factor),
    )
    info["recovery_parquet_path"] = str(recovery_parquet)
    info["recovery_camera_videos"] = recovery_camera_videos
    info["recovery_length_frames"] = recovery_length
    info["recovery_generation"] = recovery_gen

    recovery_usable = (
        bool(recovery_gen["passed"]) and not bool(recovery_gen["early_stopped"])
    )
    if not recovery_usable:
        info["case"] = CASE_RECOVERY_FAIL
        info["case_description"] = (
            f"{CASE_DESCRIPTIONS[CASE_RECOVERY_FAIL]}: "
            f"{recovery_gen['reason']}"
        )
        _write_recovery_info(scene_output_dir, info)
        return info

    if info["recovery_length_frames"] < int(prediction_horizon_frames):
        info["case"] = CASE_TOO_SHORT
        info["case_description"] = (
            f"{CASE_DESCRIPTIONS[CASE_TOO_SHORT]} "
            f"(recovery_length_frames={info['recovery_length_frames']} < "
            f"prediction_horizon_frames={int(prediction_horizon_frames)})"
        )
    elif info["recovery_length_frames"] >= info["target_recovery_frames"]:
        info["case"] = CASE_OK
        info["case_description"] = CASE_DESCRIPTIONS[CASE_OK]
    else:
        info["case"] = CASE_OK_PARTIAL
        info["case_description"] = CASE_DESCRIPTIONS[CASE_OK_PARTIAL]

    _write_recovery_info(scene_output_dir, info)
    return info


__all__ = [
    "CASE_OK",
    "CASE_OK_PARTIAL",
    "CASE_NO_DIVERGE",
    "CASE_TOO_SHORT",
    "CASE_WORKER_CRASH",
    "CASE_PLANNER_D_FAIL",
    "CASE_RECOVERY_FAIL",
    "CASE_DESCRIPTIONS",
    "generate_corrective_recovery",
    "write_policy_pass_info",
]

"""corrective worker: policy rollouts followed by stable-motion recovery collection.

Launched by scripts.policy_training_corrective. Uses the task's
required stable_solution_path for reference and corrective trajectories.
"""

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pandas as pd
import yaml

from policy.policy_training.config import SUPPORTED_POLICIES, load_policy_config


from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from policy.policy_evaluation.common import (
    run_rgb_inference,
    run_rgb_inference_batch_multiprocess,
)
from policy.policy_training.act_processors import load_act_for_rollout
from policy.policy_training.rgb.diffusion import (
    DiffusionActionWrapper,
    make_or_load_diffusion_processors,
)
from policy.policy_training.preprocessing import RGB_IMAGE_SIZE
from policy.scene_generator.generator import make_generator
from policy.evaluate import evaluate_episode, load_module, bind_helpers

from policy.corrective_data_generation.corrective import (
    CASE_OK,
    CASE_OK_PARTIAL,
    CASE_NO_DIVERGE,
    CASE_TOO_SHORT,
    CASE_WORKER_CRASH,
    CASE_PLANNER_D_FAIL,
    CASE_RECOVERY_FAIL,
    generate_corrective_recovery,
    write_policy_pass_info,
)
from policy.corrective_data_generation.utils import dense_camera_video_paths


def run_policy_rollout(
    xml_path,
    output_dir,
    file_id,
    policy,
    policy_cfg,
    eval_cfg,
    camera_names,
    device,
    log_video,
    max_ee_linear_vel,
    no_ee=False,
    dummy_simulate_t=0.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{file_id}.parquet"
    dense_paths = dense_camera_video_paths(output_dir, camera_names)
    camera_video_paths = {
        camera_name: dense_paths[f"observation.images.{camera_name}"]
        for camera_name in camera_names
    }
    run_rgb_inference(
        policy=policy,
        parquet_path=parquet_path,
        video_path=None,
        xml_path=xml_path,
        setting_cfg=policy_cfg["setting"],
        eval_cfg=eval_cfg,
        device=device,
        log_video=log_video,
        camera_names=camera_names,
        no_ee=no_ee,
        use_absolute_action=True,
        action_ensemble=True,
        max_ee_linear_vel=max_ee_linear_vel,
        camera_video_paths=camera_video_paths,
        log_video_every_step=True,
        resize_rgb_image=policy_cfg["setting"]["resize_rgb_image"],
        dummy_simulate_t=dummy_simulate_t,
    )
    return parquet_path, camera_video_paths, None


def run_policy_rollout_batch(
    rollout_jobs,
    policy,
    policy_cfg,
    eval_cfg,
    camera_names,
    device,
    log_video,
    max_ee_linear_vel,
    no_ee=False,
    dummy_simulate_t=0.0,
):
    rollout_specs = []
    for job in rollout_jobs:
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        rollout_specs.append(
            {
                "xml_path": job["xml_path"],
                "parquet_path": output_dir / f"{job['file_id']}.parquet",
                "video_path": output_dir / f"{job['file_id']}.mp4",
            }
        )

    run_rgb_inference_batch_multiprocess(
        policy=policy,
        rollout_specs=rollout_specs,
        setting_cfg=policy_cfg["setting"],
        eval_cfg=eval_cfg,
        device=device,
        log_video=log_video,
        camera_names=camera_names,
        no_ee=no_ee,
        use_absolute_action=True,
        action_ensemble=True,
        max_ee_linear_vel=max_ee_linear_vel,
        resize_rgb_image=policy_cfg["setting"]["resize_rgb_image"],
        dummy_simulate_t=dummy_simulate_t,
    )
    return [spec["parquet_path"] for spec in rollout_specs]


# =============================================================================
# Constraint evaluation wrapper (returns last_success_stage + last_success_frame)
# =============================================================================
def eval_policy_parquet(policy_parquet_path, constraint_judge_path, helper_func_path,
                        policy_tolerance):
    helpers_mod = load_module(helper_func_path, "constraint_helpers")
    constraints_judge = load_module(constraint_judge_path, "constraints_judge")
    bind_helpers(constraints_judge, helpers_mod)
    df = pd.read_parquet(policy_parquet_path)
    passed, last_success_stage, last_success_frame = evaluate_episode(
        parquet=df,
        constraints=constraints_judge,
        grasp_stage_actions=constraints_judge.grasp_stage,
        tolerance=policy_tolerance,
    )
    return {
        "status": "success" if passed else "fail",
        "last_success_stage": -1 if passed else int(last_success_stage),
        "last_success_frame": -1 if passed else int(last_success_frame),
    }


def build_rgb_act_policy(ckpt_path, camera_names, n_obs_steps, chunk_size,
                         n_action_steps, device):
    return load_act_for_rollout(ckpt_path, device)


def build_rgb_diffusion_encoder_v2_policy(
    ckpt_path,
    camera_names,
    n_obs_steps,
    chunk_size,
    n_action_steps,
    device,
):
    from lerobot.policies.diffusion_encoder_finetune_v2.modeling_diffusion import (
        DiffusionPolicy as DiffusionEncoderFinetuneV2Policy,
    )
    from lerobot.policies.diffusion_encoder_finetune_v2.configuration_diffusion import (
        DiffusionConfig as DiffusionEncoderFinetuneV2Config,
    )

    camera_keys = [f"observation.images.{name}" for name in camera_names]
    input_features = {
        camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *RGB_IMAGE_SIZE))
        for camera_key in camera_keys
    }
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(10,)
    )
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))}
    cfg = DiffusionEncoderFinetuneV2Config(
        n_obs_steps=n_obs_steps,
        horizon=chunk_size,
        n_action_steps=n_action_steps,
        input_features=input_features,
        output_features=output_features,
        device=device,
        normalization_mapping={
            "VISUAL": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.ABSOLUTE_STATE_MIN_MAX_V1,
            "ACTION": NormalizationMode.ABSOLUTE_ACTION_MIN_MAX_V1,
            "KEYPOINTS": NormalizationMode.MIN_MAX,
            "POINTCLOUD": NormalizationMode.IDENTITY,
        },
        num_inference_steps=100,
        crop_shape=None,
    )
    policy = DiffusionEncoderFinetuneV2Policy.from_pretrained(
        pretrained_name_or_path=ckpt_path,
        config=cfg,
    )
    policy.to(device)
    preprocessor, postprocessor = make_or_load_diffusion_processors(
        cfg,
        dataset_stats=None,
        pretrained_ckpt_path=ckpt_path,
    )
    return DiffusionActionWrapper(policy, preprocessor, postprocessor)


def build_rgb_openpi_policy(
    ckpt_path,
    camera_names,
    n_obs_steps,
    chunk_size,
    n_action_steps,
    device,
    policy_cfg,
    task_cfg,
    task_name,
    seed,
):
    from policy.policy_training.rgb.openpi import (
        build_openpi_eval_wrapper,
        task_camera_keys,
    )

    del camera_names, n_obs_steps, chunk_size, n_action_steps
    setting_cfg = policy_cfg["setting"]
    return build_openpi_eval_wrapper(
        setting_cfg=setting_cfg,
        checkpoint_dir=ckpt_path,
        repo_id=task_name,
        camera_keys=task_camera_keys(task_cfg),
        image_keys=tuple(setting_cfg["image_keys"]),
        task_description=str(task_cfg["task_description"]),
        device=device,
        seed=seed,
    )


_BUILD_RGB_POLICY_BY_NAME = {
    "ACT": build_rgb_act_policy,
    "diffusion_encoder_v2": build_rgb_diffusion_encoder_v2_policy,
}


def sample_scene_params(scene_gen, n_samples, rng):
    seed = int(rng.integers(0, 2**32))
    return scene_gen.sample_params(n_samples, seed)


# =============================================================================
# Main worker loop
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="corrective recovery data collection worker.")
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--target_frames", type=int, default=None,
                   help="Target recovery dataset frames to collect.")
    p.add_argument("--target_divergences", type=int, default=None,
                   help="Stop after this many valid policy divergences.")
    p.add_argument("--target_failures", type=int, default=None,
                   help="Stop after this many failed policy rollouts; skip recovery.")
    p.add_argument("--divergence_case", type=int, default=None,
                   help="Only count divergences with this recovery case.")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--rollout_dir", type=str, required=True)
    p.add_argument("--device", type=str, required=True)
    p.add_argument("--policy_cfg", type=str, required=True)
    p.add_argument("--representation", type=str, required=True)
    p.add_argument("--policy_name", choices=SUPPORTED_POLICIES, required=True)
    p.add_argument("--policy_preset", default="default")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--tasks_config_path", type=str, required=True)
    p.add_argument("--rekep_config_path", type=str,
                   default="configurations/keypoint_program_solver/config.yaml")
    p.add_argument("--recovery_seconds", type=float, required=True)
    p.add_argument("--planner_lerobot_fps", type=int, default=50)
    p.add_argument("--policy_lerobot_fps", type=int, default=10)
    p.add_argument("--policy_infer_freq", type=int, required=True)
    p.add_argument("--pos_diff_threshold", type=float, required=True)
    p.add_argument("--policy_tolerance", type=float, default=0.005)
    p.add_argument("--max_scenes", type=int, default=200)
    p.add_argument("--rollout_batch_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_video", action="store_true")
    p.add_argument("--no_ee", action="store_true",
                   help="Zero observation.state during policy rollout.")
    return p.parse_args()


def main():
    args = parse_args()
    rollout_dir = Path(args.rollout_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = rollout_dir / f"worker_{args.worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    results_path = rollout_dir / f"worker_{args.worker_id}_results.json"

    try:
        if args.representation != "rgb":
            raise NotImplementedError(
                f"representation={args.representation} is not wired up yet "
                "(supported: rgb)"
            )
        if args.rollout_batch_size <= 0:
            raise ValueError("--rollout_batch_size must be > 0")
        target_mode_count = sum(
            value is not None for value in (
                args.target_frames, args.target_divergences, args.target_failures
            )
        )
        if target_mode_count != 1:
            raise ValueError(
                "Specify exactly one of --target_frames, --target_divergences, "
                "or --target_failures"
            )
        if args.target_frames is not None and args.target_frames <= 0:
            raise ValueError("--target_frames must be > 0")
        if args.target_divergences is not None and args.target_divergences <= 0:
            raise ValueError("--target_divergences must be > 0")
        if args.target_failures is not None and args.target_failures <= 0:
            raise ValueError("--target_failures must be > 0")

        policy_cfg = load_policy_config(args.policy_cfg, args.policy_name, args.policy_preset)
        with open(args.tasks_config_path, "r") as f:
            tasks_full = yaml.safe_load(f)
        tasks_cfg = tasks_full["tasks"]
        task_cfg = tasks_cfg[args.task]
        dummy_simulate_t = float(task_cfg["dummy_simulate_t"])
        max_ee_linear_vel = float(task_cfg["max_ee_linear_vel"])
        eval_cfg = task_cfg["eval"]

        chunk_size = policy_cfg["setting"]["chunk_size"]
        n_action_steps = policy_cfg["setting"]["n_action_steps"]
        n_obs_steps = policy_cfg["setting"]["n_obs_steps"]
        camera_names = task_cfg["camera_names"]

        stable_solution_path = os.path.abspath(
            os.path.join(".", task_cfg["stable_solution_path"])
        )
        constraint_judge_path = os.path.abspath(
            os.path.join(".", task_cfg["constraint_judge_path"])
        )
        helper_func_path = os.path.abspath(
            os.path.join(".", "keypoint_program_solver/constraint_helpers.py")
        )

        if args.policy_name == "Pi05":
            policy = build_rgb_openpi_policy(
                ckpt_path=args.ckpt_path,
                camera_names=camera_names,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                device=args.device,
                policy_cfg=policy_cfg,
                task_cfg=task_cfg,
                task_name=args.task,
                seed=args.seed,
            )
        else:
            policy = _BUILD_RGB_POLICY_BY_NAME[args.policy_name](
                ckpt_path=args.ckpt_path,
                camera_names=camera_names,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                device=args.device,
            )

        scene_gen = make_generator(args.task, tasks_full)

        rng = np.random.default_rng(args.seed + args.worker_id * 10_000)

        rekep_tolerance = float(task_cfg["rekep_tolerance"])
        t_horizon = float(task_cfg["eval"]["t_horizon"])

        accumulated = 0
        divergence_count = 0
        divergence_records = []
        case_counts = {
            CASE_OK: 0,
            CASE_OK_PARTIAL: 0,
            CASE_NO_DIVERGE: 0,
            CASE_TOO_SHORT: 0,
            CASE_PLANNER_D_FAIL: 0,
            CASE_RECOVERY_FAIL: 0,
        }
        policy_passed_count = 0
        policy_evaluated_count = 0
        policy_failed_count = 0
        failure_records = []
        recovery_skipped_after_target_count = 0
        collected_parquets = []
        collected_records = []
        scene_idx = 0

        if args.planner_lerobot_fps <= 0 or args.policy_lerobot_fps <= 0 \
                or args.policy_infer_freq <= 0:
            raise ValueError(
                "--planner_lerobot_fps, --policy_lerobot_fps, "
                "and --policy_infer_freq must all be > 0."
            )
        if args.planner_lerobot_fps % args.policy_lerobot_fps != 0:
            raise ValueError(
                f"planner_lerobot_fps ({args.planner_lerobot_fps}) must be a "
                f"multiple of policy_lerobot_fps ({args.policy_lerobot_fps})."
            )
        policy2planner_factor = int(args.planner_lerobot_fps / args.policy_lerobot_fps)
        target_recovery_frames = int(
            round(args.recovery_seconds * args.planner_lerobot_fps)
        )

        prediction_horizon_frames = int(
            chunk_size / args.policy_infer_freq * args.planner_lerobot_fps
        )

        assert prediction_horizon_frames == 80 # assume for sanity check

        use_batched_policy_rollout = (
            args.policy_name in SUPPORTED_POLICIES
            and args.rollout_batch_size > 1
        )
        rollout_mode = "batched" if use_batched_policy_rollout else "single"
        print(
            f"[worker {args.worker_id}] Policy rollout mode: {rollout_mode} "
            f"(rollout_batch_size={args.rollout_batch_size})"
        )

        def target_met():
            if args.target_failures is not None:
                return policy_failed_count >= args.target_failures
            if args.target_divergences is not None:
                return divergence_count >= args.target_divergences
            return accumulated >= args.target_frames

        while not target_met() and scene_idx < args.max_scenes:
            batch_size = args.rollout_batch_size if use_batched_policy_rollout else 1
            scene_batch_size = min(batch_size, args.max_scenes - scene_idx)
            if args.target_failures is not None:
                scene_batch_size = min(
                    scene_batch_size,
                    args.target_failures - policy_failed_count,
                )
            scene_params_batch = sample_scene_params(scene_gen, scene_batch_size, rng)
            rollout_jobs = []
            while len(rollout_jobs) < scene_batch_size:
                file_id = f"sample_{scene_idx}"
                scene_dir = worker_dir / file_id
                scene_dir.mkdir(parents=True, exist_ok=True)
                policy_dir = scene_dir / "policy_rollout"

                scene_params = scene_params_batch[len(rollout_jobs)]
                xml_path = scene_dir / "scene/sample_0/tmp_scene.xml"
                if not xml_path.is_file():
                    xml_path = scene_gen.generate_scene(scene_params, scene_dir / "scene", 0)
                rollout_jobs.append(
                    {
                        "file_id": file_id,
                        "scene_dir": scene_dir,
                        "output_dir": policy_dir,
                        "xml_path": os.path.abspath(xml_path),
                    }
                )
                scene_idx += 1

            if use_batched_policy_rollout:
                try:
                    policy_parquets = run_policy_rollout_batch(
                        rollout_jobs=rollout_jobs,
                        policy=policy,
                        policy_cfg=policy_cfg,
                        eval_cfg=eval_cfg,
                        camera_names=camera_names,
                        device=args.device,
                        log_video=args.log_video,
                        max_ee_linear_vel=max_ee_linear_vel,
                        no_ee=args.no_ee,
                        dummy_simulate_t=dummy_simulate_t,
                    )
                except Exception:
                    for job in rollout_jobs:
                        err_path = job["scene_dir"] / "policy_rollout_error.txt"
                        with open(err_path, "w") as f:
                            f.write(traceback.format_exc())
                    gc.collect()
                    continue
                for job, policy_parquet in zip(rollout_jobs, policy_parquets):
                    job["policy_parquet"] = policy_parquet
            else:
                job = rollout_jobs[0]
                try:
                    policy_parquet, _, _ = run_policy_rollout(
                        xml_path=job["xml_path"],
                        output_dir=job["output_dir"],
                        file_id=job["file_id"],
                        policy=policy,
                        policy_cfg=policy_cfg,
                        eval_cfg=eval_cfg,
                        camera_names=camera_names,
                        device=args.device,
                        log_video=args.log_video,
                        max_ee_linear_vel=max_ee_linear_vel,
                        no_ee=args.no_ee,
                        dummy_simulate_t=dummy_simulate_t,
                    )
                    job["policy_parquet"] = policy_parquet
                except Exception:
                    err_path = job["scene_dir"] / "policy_rollout_error.txt"
                    with open(err_path, "w") as f:
                        f.write(traceback.format_exc())
                    gc.collect()
                    continue

            failed_jobs = []
            for job in rollout_jobs:
                policy_parquet = job["policy_parquet"]
                eval_result = eval_policy_parquet(
                    policy_parquet, constraint_judge_path, helper_func_path,
                    task_cfg["policy_tolerance"],
                )
                job["eval_result"] = eval_result
                policy_evaluated_count += 1

                if eval_result["status"] == "success":
                    write_policy_pass_info(
                        scene_output_dir=job["scene_dir"],
                        xml_path=job["xml_path"],
                        policy_parquet_path=policy_parquet,
                        representation=args.representation,
                        recovery_seconds=args.recovery_seconds,
                        planner_fps=args.planner_lerobot_fps,
                    )
                    policy_passed_count += 1
                    gc.collect()
                    continue

                policy_failed_count += 1
                failed_jobs.append(job)
                failure_records.append({
                    "sample_id": job["file_id"],
                    "policy_parquet_path": str(policy_parquet),
                    "last_success_stage": int(eval_result["last_success_stage"]),
                    "last_success_frame": int(eval_result["last_success_frame"]),
                })

            if args.target_failures is not None:
                gc.collect()
                continue

            for failed_job_idx, job in enumerate(failed_jobs):
                if target_met():
                    recovery_skipped_after_target_count += (
                        len(failed_jobs) - failed_job_idx
                    )
                    break

                eval_result = job["eval_result"]
                policy_parquet = job["policy_parquet"]
                try:
                    info = generate_corrective_recovery(
                        xml_path=job["xml_path"],
                        policy_parquet_path=policy_parquet,
                        scene_output_dir=job["scene_dir"],
                        representation=args.representation,
                        stable_solution_path=stable_solution_path,
                        constraint_judge_path=constraint_judge_path,
                        helper_func_path=helper_func_path,
                        rekep_config_path=args.rekep_config_path,
                        pos_diff_threshold=args.pos_diff_threshold,
                        policy2planner_factor=policy2planner_factor,
                        recovery_seconds=args.recovery_seconds,
                        planner_fps=args.planner_lerobot_fps,
                        prediction_horizon_frames=prediction_horizon_frames,
                        t_horizon=t_horizon,
                        rekep_tolerance=rekep_tolerance,
                        max_ee_linear_vel=max_ee_linear_vel,
                        file_id=job["file_id"],
                        last_success_stage=eval_result["last_success_stage"],
                        last_success_frame=eval_result["last_success_frame"],
                    )
                except Exception:
                    err_path = job["scene_dir"] / "recovery_error.txt"
                    with open(err_path, "w") as f:
                        f.write(traceback.format_exc())
                    gc.collect()
                    continue

                case = info["case"]
                case_counts[case] = case_counts.get(case, 0) + 1
                recovery_parquet_path = info.get("recovery_parquet_path")

                policy_diverge_idx = info.get("policy_diverge_idx")
                is_valid_divergence = (
                    bool(info.get("diverged", False))
                    and isinstance(policy_diverge_idx, int)
                    and not isinstance(policy_diverge_idx, bool)
                    and policy_diverge_idx >= 0
                    and (
                        args.divergence_case is None
                        or case == args.divergence_case
                    )
                )
                if is_valid_divergence:
                    divergence_count += 1
                    divergence_records.append({
                        "sample_id": job["file_id"],
                        "case": int(case),
                        "policy_diverge_idx": int(policy_diverge_idx),
                        "planner_d_diverge_idx": int(
                            info.get("planner_d_diverge_idx", -1)
                        ),
                        "stage_d_at_diverge": info.get("stage_d_at_diverge"),
                        "recovery_info_path": str(
                            job["scene_dir"] / "recovery_info.json"
                        ),
                        "policy_parquet_path": str(policy_parquet),
                    })

                if case in (CASE_OK, CASE_OK_PARTIAL) and recovery_parquet_path:
                    collected_parquets.append(recovery_parquet_path)
                    if args.representation == "rgb":
                        collected_records.append({
                            "recovery_parquet": recovery_parquet_path,
                            "xml": info["xml_path"],
                            "camera_videos": info["recovery_camera_videos"],
                        })
                    clipped_frames = min(
                        target_recovery_frames,
                        info["recovery_length_frames"],
                    )
                    assert clipped_frames >= prediction_horizon_frames
                    accumulated += clipped_frames

                gc.collect()

        results = {
            "worker_id": args.worker_id,
            "device": args.device,
            "target_frames": args.target_frames,
            "accumulated_frames": int(accumulated),
            "target_divergences": args.target_divergences,
            "target_failures": args.target_failures,
            "divergence_case": args.divergence_case,
            "divergence_count": int(divergence_count),
            "divergence_records": divergence_records,
            "target_recovery_frames": target_recovery_frames,
            "scenes_generated": scene_idx,
            "policy_evaluated_count": policy_evaluated_count,
            "policy_passed_count": policy_passed_count,
            "policy_failed_count": policy_failed_count,
            "failure_records": failure_records,
            "policy_success_rate": (
                policy_passed_count / policy_evaluated_count
                if policy_evaluated_count > 0 else 0.0
            ),
            "recovery_attempted_count": sum(case_counts.values()),
            "recovery_skipped_after_target_count": recovery_skipped_after_target_count,
            "case_counts": {str(k): v for k, v in case_counts.items()},
            "collected_recovery_parquets": collected_parquets,
            "collected_recovery_records": collected_records,
            "case": (
                "target_met" if target_met() else "exhausted_max_scenes"
            ),
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    except Exception:
        err_payload = {
            "worker_id": args.worker_id,
            "case": CASE_WORKER_CRASH,
            "case_description": "worker crashed",
            "traceback": traceback.format_exc(),
        }
        with open(results_path, "w") as f:
            json.dump(err_payload, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()

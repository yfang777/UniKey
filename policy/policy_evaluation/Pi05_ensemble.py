from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from policy.policy_training.config import load_policy_config

from policy.policy_evaluation.common import (
    named_scenes,
    run_rgb_inference,
    run_rgb_inference_batch,
    run_rgb_inference_batch_multiprocess,
)
from policy.policy_training.rgb.openpi import (
    OPENPI_IMAGE_KEYS,
    build_openpi_eval_wrapper,
    normalize_openpi_params_path,
    task_camera_keys,
)
from policy.evaluate import evaluate_rollout


POLICY_KEY = "Pi05"




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Codegen RGB policy with JAX OpenPI pi0.5.")
    parser.add_argument("--xml_list", type=str, required=True, nargs="+")
    parser.add_argument("--policy_config_path", type=str, required=True)
    parser.add_argument("--policy_preset", default="default")
    parser.add_argument("--pretrained_ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_video", action="store_true")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--tasks_config_path", type=str, default="configurations/tasks.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ee", action="store_true")
    parser.add_argument("--use_absolute_action", action="store_true")
    parser.add_argument("--rollout_batch_size", type=int, default=1)
    parser.add_argument("--multiprocess_envs", action="store_true")
    parser.add_argument(
        "--no_action_ensemble",
        action="store_false",
        dest="action_ensemble",
    )
    parser.set_defaults(action_ensemble=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_dir_from_arg(path: str | Path) -> Path:
    params_path = Path(normalize_openpi_params_path(path))
    if params_path.name == "params":
        return params_path.parent
    if (params_path / "params").exists():
        return params_path
    if (params_path / "final" / "params").exists():
        return params_path / "final"
    raise FileNotFoundError(f"OpenPI checkpoint does not contain params: {path}")


def rollout_batch(args: argparse.Namespace) -> None:
    if not args.use_absolute_action:
        raise ValueError("Pi05 evaluation is wired for absolute actions; pass --use_absolute_action.")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout_batch_size must be > 0")

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_policy_config(args.policy_config_path, POLICY_KEY, args.policy_preset)

    with open(args.tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)["tasks"]

    setting_cfg = config["setting"]
    task_cfg = tasks_config[args.task]
    task_description = str(task_cfg["task_description"])
    eval_cfg = task_cfg["eval"]
    max_ee_linear_vel = float(task_cfg["max_ee_linear_vel"])
    camera_names = task_cfg["camera_names"]
    camera_keys = task_camera_keys(task_cfg)
    image_keys = tuple(setting_cfg.get("image_keys", OPENPI_IMAGE_KEYS))
    checkpoint_dir = checkpoint_dir_from_arg(args.pretrained_ckpt_path)

    policy = build_openpi_eval_wrapper(
        setting_cfg=setting_cfg,
        checkpoint_dir=checkpoint_dir,
        repo_id=args.task,
        camera_keys=camera_keys,
        image_keys=image_keys,
        task_description=task_description,
        device=args.device,
        seed=args.seed,
    )

    constraint_path = os.path.abspath(task_cfg["constraint_judge_path"])
    helper_func_path = os.path.abspath("keypoint_program_solver/constraint_helpers.py")
    tolerance_value = task_cfg["policy_tolerance"]

    rollout_info = []
    scenes = named_scenes(args.xml_list)
    if args.rollout_batch_size <= 1:
        for xml_path, file_id in scenes:
            video_path = output_dir / f"{file_id}.mp4"
            parquet_path = output_dir / f"{file_id}.parquet"

            run_rgb_inference(
                policy=policy,
                parquet_path=parquet_path,
                video_path=video_path,
                xml_path=xml_path,
                setting_cfg=setting_cfg,
                eval_cfg=eval_cfg,
                device=args.device,
                log_video=args.log_video,
                camera_names=camera_names,
                no_ee=args.no_ee,
                use_absolute_action=True,
                action_ensemble=args.action_ensemble,
                max_ee_linear_vel=max_ee_linear_vel,
                resize_rgb_image=False,
            )

            rollout_info.append(
                {
                    "xml_path": str(xml_path),
                    "video_path": str(video_path) if args.log_video else "Video Not Logged",
                    "parquet_path": str(parquet_path),
                }
            )
    else:
        for start in range(0, len(args.xml_list), args.rollout_batch_size):
            xml_chunk = scenes[start : start + args.rollout_batch_size]
            rollout_specs = []
            for xml_path, file_id in xml_chunk:
                video_path = output_dir / f"{file_id}.mp4"
                parquet_path = output_dir / f"{file_id}.parquet"
                rollout_specs.append(
                    {
                        "xml_path": xml_path,
                        "video_path": video_path,
                        "parquet_path": parquet_path,
                    }
                )

            if args.multiprocess_envs:
                run_rgb_inference_batch_multiprocess(
                    policy=policy,
                    rollout_specs=rollout_specs,
                    setting_cfg=setting_cfg,
                    eval_cfg=eval_cfg,
                    device=args.device,
                    log_video=args.log_video,
                    camera_names=camera_names,
                    no_ee=args.no_ee,
                    use_absolute_action=True,
                    action_ensemble=args.action_ensemble,
                    max_ee_linear_vel=max_ee_linear_vel,
                    resize_rgb_image=False,
                )
            else:
                run_rgb_inference_batch(
                    policy=policy,
                    rollout_specs=rollout_specs,
                    setting_cfg=setting_cfg,
                    eval_cfg=eval_cfg,
                    device=args.device,
                    log_video=args.log_video,
                    camera_names=camera_names,
                    no_ee=args.no_ee,
                    use_absolute_action=True,
                    action_ensemble=args.action_ensemble,
                    max_ee_linear_vel=max_ee_linear_vel,
                    resize_rgb_image=False,
                )

            for rollout_spec in rollout_specs:
                rollout_info.append(
                    {
                        "xml_path": str(rollout_spec["xml_path"]),
                        "video_path": str(rollout_spec["video_path"]) if args.log_video else "Video Not Logged",
                        "parquet_path": str(rollout_spec["parquet_path"]),
                    }
                )

    output_json_path = output_dir / "info.json"
    with open(output_json_path, "w") as f:
        json.dump(rollout_info, f, indent=2)
    evaluate_rollout(output_json_path, constraint_path, helper_func_path, tolerance_value)


if __name__ == "__main__":
    rollout_batch(parse_args())

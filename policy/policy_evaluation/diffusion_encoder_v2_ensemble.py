import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion_encoder_finetune_v2.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion_encoder_finetune_v2.modeling_diffusion import DiffusionPolicy

from policy.policy_evaluation.common import (
    named_scenes,
    run_rgb_inference,
    run_rgb_inference_batch,
    run_rgb_inference_batch_multiprocess,
)
from policy.policy_training.rgb.diffusion import (
    DiffusionActionWrapper,
    make_or_load_diffusion_processors,
)
from policy.policy_training.preprocessing import RGB_IMAGE_SIZE
from policy.evaluate import evaluate_rollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_list", type=str, required=True, nargs="+")
    parser.add_argument("--policy_config_path", type=str, required=True)
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_batch(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.policy_config_path, "r") as f:
        config = yaml.safe_load(f)["diffusion_encoder_v2"]

    with open(args.tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)["tasks"]

    task_cfg = tasks_config[args.task]
    dummy_simulate_t = float(task_cfg["dummy_simulate_t"])
    max_ee_linear_vel = float(task_cfg["max_ee_linear_vel"])
    setting_cfg = config["setting"]
    eval_cfg = task_cfg["eval"]
    chunk_size = int(setting_cfg["chunk_size"])
    n_action_steps = int(setting_cfg["n_action_steps"])
    n_obs_steps = int(setting_cfg["n_obs_steps"])

    camera_names = task_cfg["camera_names"]
    camera_keys = [
        f"observation.images.{camera_name}"
        for camera_name in camera_names
    ]

    input_features = {
        camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *RGB_IMAGE_SIZE))
        for camera_key in camera_keys
    }
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(10,))
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))}

    policy_config = DiffusionConfig(
        n_obs_steps=n_obs_steps,
        horizon=chunk_size,
        n_action_steps=n_action_steps,
        input_features=input_features,
        output_features=output_features,
        num_inference_steps=100,
        crop_shape=None,
        device=args.device,
        normalization_mapping={
            "VISUAL": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.ABSOLUTE_STATE_MIN_MAX_V1,
            "ACTION": (NormalizationMode.RELATIVE_ACTION if not args.use_absolute_action
                       else NormalizationMode.ABSOLUTE_ACTION_MIN_MAX_V1),
        },
    )

    policy = DiffusionPolicy.from_pretrained(
        pretrained_name_or_path=args.pretrained_ckpt_path,
        config=policy_config,
    )
    policy.to(args.device)
    preprocessor, postprocessor = make_or_load_diffusion_processors(
        policy_config,
        dataset_stats=None,
        pretrained_ckpt_path=args.pretrained_ckpt_path,
    )
    policy = DiffusionActionWrapper(policy, preprocessor, postprocessor)

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
                use_absolute_action=args.use_absolute_action,
                action_ensemble=args.action_ensemble,
                max_ee_linear_vel=max_ee_linear_vel,
                resize_rgb_image=True,
                dummy_simulate_t=dummy_simulate_t,
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
            xml_chunk = scenes[start:start + args.rollout_batch_size]
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
                    use_absolute_action=args.use_absolute_action,
                    action_ensemble=args.action_ensemble,
                    max_ee_linear_vel=max_ee_linear_vel,
                    resize_rgb_image=True,
                    dummy_simulate_t=dummy_simulate_t,
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
                    use_absolute_action=args.use_absolute_action,
                    action_ensemble=args.action_ensemble,
                    max_ee_linear_vel=max_ee_linear_vel,
                    resize_rgb_image=True,
                    dummy_simulate_t=dummy_simulate_t,
                )

            for rollout_spec in rollout_specs:
                rollout_info.append(
                    {
                        "xml_path": str(rollout_spec["xml_path"]),
                        "video_path": (
                            str(rollout_spec["video_path"])
                            if args.log_video
                            else "Video Not Logged"
                        ),
                        "parquet_path": str(rollout_spec["parquet_path"]),
                    }
                )

    output_json_path = output_dir / "info.json"
    with open(output_json_path, "w") as f:
        json.dump(rollout_info, f, indent=2)
    evaluate_rollout(output_json_path, constraint_path, helper_func_path, tolerance_value)


if __name__ == "__main__":
    rollout_batch(parse_args())

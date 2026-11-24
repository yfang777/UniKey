import argparse
import json
import os
from pathlib import Path

import yaml

from policy.policy_evaluation.common import (
    named_scenes,
    run_rgb_inference,
    run_rgb_inference_batch,
    run_rgb_inference_batch_multiprocess,
)
from policy.policy_training.act_processors import load_act_for_rollout
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


def rollout_batch(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.policy_config_path, "r") as f:
        config = yaml.safe_load(f)["ACT"]

    with open(args.tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)["tasks"]

    task_cfg = tasks_config[args.task]
    dummy_simulate_t = float(task_cfg["dummy_simulate_t"])
    max_ee_linear_vel = float(task_cfg["max_ee_linear_vel"])
    setting_cfg = config["setting"]
    eval_cfg = task_cfg["eval"]

    camera_names = task_cfg["camera_names"]
    policy = load_act_for_rollout(args.pretrained_ckpt_path, args.device)

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

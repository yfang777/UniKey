import argparse
import gc
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from lerobot.configs.types import NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from tqdm import tqdm

from policy.policy_training.rgb.dataset_combined_v4 import (
    CombinedRgbLeRobotDatasetV4,
    RatioBatchSampler,
)
from policy.policy_training.act_processors import configure_absolute_actions, make_or_load_act_processors, save_act_checkpoint
from policy.policy_training.preprocessing import RGB_IMAGE_SIZE, resize_rgb_observation
from policy.policy_training.utils import abs_to_rel_actions, inject_noise, update_relative_action_stats
from keypoint_program_solver.ik import KinHelper
from keypoint_program_solver.utils import resolve_robot_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--recovery_dataset_path", type=str, default=None)
    parser.add_argument("--new_recovery_episode_count", type=int, default=0)
    parser.add_argument("--policy_config_path", type=str, required=True)
    parser.add_argument("--pretrained_ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--tasks_config_path", type=str, default="configurations/tasks.yaml")
    parser.add_argument("--no_ee", action="store_true")
    parser.add_argument("--use_absolute_action", action="store_true")
    parser.add_argument("--add_action_noise", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    with open(args.policy_config_path, "r") as f:
        config = yaml.safe_load(f)["ACT"]

    with open(args.tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)["tasks"]
    task_cfg = tasks_config[args.task]

    wandb.init(
        project="ACT_training",
        entity=None,
        config=config,
        dir=os.environ.get("WANDB_DIR", os.path.expanduser("~/.cache/wandb")),
        name=f"ACT_rgb_combined_{Path(args.ckpt_save_dir).parent.name}_{Path(args.ckpt_save_dir).name}",
    )

    repo_id = config["setting"]["repo_id"]
    policy_infer_freq = int(config["setting"]["policy_infer_freq"])
    action_timestamp_offset = float(config["setting"]["action_timestamp_offset"])
    n_obs_steps = int(config["setting"]["n_obs_steps"])
    chunk_size = int(config["setting"]["chunk_size"])
    n_action_steps = int(config["setting"]["n_action_steps"])
    use_relative_action = not args.use_absolute_action
    batch_size = int(config["train"]["batch_size_rgb"])
    num_workers = int(config["train"]["num_workers_rgb"])
    dataloader_recycle_steps = int(config["train"]["dataloader_recycle_steps_rgb"])
    if dataloader_recycle_steps < 0:
        raise ValueError("dataloader_recycle_steps_rgb must be >= 0")
    dataset_ratio_cfg = config["train"]["dataset_combined_v4"]
    camera_keys = [
        f"observation.images.{camera_name}"
        for camera_name in task_cfg["camera_names"]
    ]

    if args.seed is not None:
        set_seed(int(args.seed))

    dataset_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=args.dataset_dir)
    dataset_stats = dataset_metadata.stats
    dataset_features = dataset_to_policy_features(dataset_metadata.features)

    input_feature_cols = camera_keys + ["observation.state"]
    output_feature_cols = ["action"]
    input_features = {
        **{
            camera_key: PolicyFeature(
                type=dataset_features[camera_key].type,
                shape=(3, *RGB_IMAGE_SIZE),
            )
            for camera_key in camera_keys
        },
        "observation.state": dataset_features["observation.state"],
    }
    output_features = {key: dataset_features[key] for key in output_feature_cols}

    act_cfg = ACTConfig(
        n_obs_steps=n_obs_steps,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=None,
        input_features=input_features,
        output_features=output_features,
    )
    act_cfg.device = args.device

    delta_timestamps = {
        "observation.state": [i / policy_infer_freq for i in range(1 - n_obs_steps, 1)],
        "action": [
            i / policy_infer_freq + action_timestamp_offset
            for i in range(chunk_size)
        ],
    }
    for camera_key in camera_keys:
        delta_timestamps[camera_key] = [
            i / policy_infer_freq for i in range(1 - n_obs_steps, 1)
        ]

    required_columns = camera_keys + [
        "observation.state",
        "action",
        "episode_index",
        "task_index",
    ]
    dataset = CombinedRgbLeRobotDatasetV4(
        required_columns=required_columns,
        recovery_dataset_path=args.recovery_dataset_path,
        new_recovery_episode_count=args.new_recovery_episode_count,
        repo_id=repo_id,
        root=args.dataset_dir,
        delta_timestamps=delta_timestamps,
        video_backend=None,
        image_transforms=None,
    )
    batch_sampler = RatioBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        ratios={
            "offline": float(dataset_ratio_cfg["offline"]),
            "new_recovery": float(dataset_ratio_cfg["new_recovery"]),
        },
        seed=int(args.seed),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        pin_memory=False,
    )

    if use_relative_action:
        dataloader_normal = torch.utils.data.DataLoader(
            dataset.base_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=len(dataset.base_dataset) >= batch_size,
        )
        update_relative_action_stats(
            stats=dataset_stats,
            horizon=chunk_size,
            dataloader=dataloader_normal,
        )
        act_cfg.normalization_mapping["ACTION"] = NormalizationMode.RELATIVE_ACTION
        raise ValueError("Need Implementation")
    else:
        dataset_stats = configure_absolute_actions(act_cfg, dataset_stats)

    preprocessor, postprocessor = make_or_load_act_processors(
        act_cfg,
        dataset_stats,
        args.pretrained_ckpt_path,
    )

    if args.pretrained_ckpt_path is not None:
        policy = ACTPolicy.from_pretrained(
            pretrained_name_or_path=args.pretrained_ckpt_path,
            config=act_cfg,
        )
        num_steps = int(config["train"]["num_steps_finetune"])
        save_freq = int(config["train"]["save_freq_finetune"])
    else:
        policy = ACTPolicy(config=act_cfg)
        num_steps = int(config["train"]["num_steps_rgb"])
        save_freq = int(config["train"]["save_freq"])
    policy.to(args.device)
    policy.train()

    optimizer_cfg = act_cfg.get_optimizer_preset()
    optimizer = optimizer_cfg.build(policy.get_optim_params())

    ckpt_dir = Path(args.ckpt_save_dir)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    kin_helper = KinHelper(resolve_robot_paths(task_cfg["robot_variant"])[0]) if args.add_action_noise else None
    progress_bar = tqdm(range(num_steps), desc="Training Steps")
    global_step = 0
    if dataloader_recycle_steps > 0:
        print(
            "Recycling RGB DataLoader workers every "
            f"{dataloader_recycle_steps} training steps."
        )

    while global_step < num_steps:
        dataloader_iter = iter(dataloader)
        recycled_steps = 0
        try:
            while global_step < num_steps and (
                dataloader_recycle_steps == 0
                or recycled_steps < dataloader_recycle_steps
            ):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break

                if args.add_action_noise:
                    batch = inject_noise(kin_helper, batch)
                if use_relative_action:
                    batch = abs_to_rel_actions(batch)

                batch = {
                    key: (
                        value.to(args.device)
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in batch.items()
                    if key in input_feature_cols or key == "action_is_pad" or key == "action"
                }
                batch = resize_rgb_observation(batch, camera_keys)

                for camera_key in camera_keys:
                    batch[camera_key] = batch[camera_key].squeeze(1)
                batch["observation.state"] = batch["observation.state"].squeeze(1)
                if args.no_ee:
                    batch["observation.state"] = torch.zeros_like(batch["observation.state"])

                batch = preprocessor(batch)
                batch["action"][..., :3].clamp_(-1.2, 1.2)
                loss, loss_dict = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                recycled_steps += 1
                log_data = {"train/loss": loss.item(), "global_step": global_step}
                if loss_dict:
                    for key, value in loss_dict.items():
                        log_data[f"train/{key}"] = (
                            value.item() if isinstance(value, torch.Tensor) else value
                        )
                wandb.log(log_data)
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % save_freq == 0:
                    save_act_checkpoint(
                        policy,
                        preprocessor,
                        postprocessor,
                        ckpt_dir / f"step_{global_step}",
                    )
        finally:
            del dataloader_iter
            gc.collect()

    print("Training finished. Saving final checkpoint...")
    save_act_checkpoint(policy, preprocessor, postprocessor, ckpt_dir / "final")
    progress_bar.close()
    wandb.finish()


if __name__ == "__main__":
    main(parse_args())

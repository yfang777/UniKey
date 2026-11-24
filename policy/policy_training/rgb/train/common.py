import copy
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from lerobot.configs.types import PolicyFeature
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from policy.policy_training.rgb.dataset import RgbLeRobotDataset
from policy.policy_training.rgb.dataset_combined_v4 import (
    CombinedRgbLeRobotDatasetV4,
    RatioBatchSampler,
)
from policy.policy_training.preprocessing import RGB_IMAGE_SIZE, resize_rgb_observation


class EMAModel:
    def __init__(
        self,
        model,
        update_after_step,
        inv_gamma,
        power,
        min_value,
        max_value,
    ):
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, model):
        self.decay = self.get_decay(self.optimization_step)

        for (ema_name, ema_param), (name, param) in zip(
            self.averaged_model.named_parameters(),
            model.named_parameters(),
            strict=True,
        ):
            assert ema_name == name
            if not param.requires_grad:
                ema_param.copy_(param.to(dtype=ema_param.dtype).data)
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(
                    param.to(dtype=ema_param.dtype).data,
                    alpha=1 - self.decay,
                )

        for (ema_name, ema_buffer), (name, buffer) in zip(
            self.averaged_model.named_buffers(),
            model.named_buffers(),
            strict=True,
        ):
            assert ema_name == name
            ema_buffer.copy_(buffer.to(dtype=ema_buffer.dtype))

        self.optimization_step += 1


def make_ema_model(policy, ema_cfg):
    if ema_cfg is False:
        return None

    ema_policy = copy.deepcopy(policy)
    return EMAModel(
        model=ema_policy,
        update_after_step=ema_cfg["update_after_step"],
        inv_gamma=ema_cfg["inv_gamma"],
        power=ema_cfg["power"],
        min_value=ema_cfg["min_value"],
        max_value=ema_cfg["max_value"],
    )


def parse_rgb_train_args(parser):
    parser.add_argument("--dataset_dir", type=str, required=True)
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
    return parser


def parse_combined_rgb_train_args(parser):
    parse_rgb_train_args(parser)
    parser.add_argument("--recovery_dataset_path", type=str, default=None)
    parser.add_argument("--new_recovery_episode_count", type=int, default=0)
    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_policy_and_task_config(policy_config_path, tasks_config_path, policy_key, task):
    with open(policy_config_path, "r") as f:
        config = yaml.safe_load(f)[policy_key]

    with open(tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)["tasks"]
    return config, tasks_config[task]


def init_wandb_run(project, config, run_name):
    wandb.init(
        project=project,
        entity=None,
        config=config,
        dir=os.environ["WANDB_DIR"] if "WANDB_DIR" in os.environ else os.path.expanduser("~/.cache/wandb"),
        name=run_name,
    )


def rgb_camera_keys(task_cfg):
    return [
        f"observation.images.{camera_name}"
        for camera_name in task_cfg["camera_names"]
    ]


def load_rgb_dataset_stats_and_features(repo_id, dataset_dir):
    dataset_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_dir)
    return dataset_metadata.stats, dataset_to_policy_features(dataset_metadata.features)


def rgb_policy_features(dataset_features, camera_keys):
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
    return input_feature_cols, input_features, output_features


def rgb_delta_timestamps(
    policy_infer_freq,
    n_obs_steps,
    chunk_size,
    action_timestamp_offset,
    camera_keys,
):
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
    return delta_timestamps


def rgb_required_columns(camera_keys):
    return camera_keys + [
        "observation.state",
        "action",
        "episode_index",
        "task_index",
    ]


def build_rgb_dataloader(
    required_columns,
    repo_id,
    dataset_dir,
    delta_timestamps,
    batch_size,
    num_workers,
):
    dataset = RgbLeRobotDataset(
        required_columns=required_columns,
        repo_id=repo_id,
        root=dataset_dir,
        delta_timestamps=delta_timestamps,
        video_backend=None,
        image_transforms=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=len(dataset) >= batch_size,
    )
    return dataset, dataloader


def build_combined_v4_rgb_dataloader(
    required_columns,
    repo_id,
    dataset_dir,
    recovery_dataset_path,
    new_recovery_episode_count,
    delta_timestamps,
    batch_size,
    num_workers,
    dataset_ratio_cfg,
    seed,
):
    dataset = CombinedRgbLeRobotDatasetV4(
        required_columns=required_columns,
        recovery_dataset_path=recovery_dataset_path,
        new_recovery_episode_count=new_recovery_episode_count,
        repo_id=repo_id,
        root=dataset_dir,
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
        seed=seed,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        pin_memory=False,
    )
    return dataset, dataloader


def prepare_rgb_batch(
    batch,
    device,
    input_feature_cols,
    camera_keys,
    no_ee,
    add_single_observation_step,
    squeeze_single_observation_step,
):
    batch = {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
        if key in input_feature_cols or key == "action_is_pad" or key == "action"
    }
    if add_single_observation_step:
        for camera_key in camera_keys:
            batch[camera_key] = batch[camera_key].unsqueeze(1)
    batch = resize_rgb_observation(batch, camera_keys)
    if squeeze_single_observation_step:
        for camera_key in camera_keys:
            batch[camera_key] = batch[camera_key].squeeze(1)
        batch["observation.state"] = batch["observation.state"].squeeze(1)
    if no_ee:
        batch["observation.state"] = torch.zeros_like(batch["observation.state"])
    return batch


def reset_checkpoint_dir(ckpt_save_dir):
    ckpt_dir = Path(ckpt_save_dir)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def log_train_loss(loss, loss_dict, global_step):
    log_data = {"train/loss": loss.item(), "global_step": global_step}
    if loss_dict:
        for key, value in loss_dict.items():
            log_data[f"train/{key}"] = (
                value.item() if isinstance(value, torch.Tensor) else value
            )
    wandb.log(log_data)

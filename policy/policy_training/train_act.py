"""Train RGB ACT on local demonstrations, optionally mixing in corrective episodes.

Adapted from Codegen_Datagen's ACT.py and ACT_combined_v3.py.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, PolicyFeature
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader

from .act_processors import make_or_load_act_processors, save_act_checkpoint
from .preprocessing import RGB_IMAGE_SIZE, resize_rgb_observation


def training_batches(dataset, batch_size):
    # Repeat by epoch without caching decoded frames. Sampling with replacement
    # also gives a full batch for short corrective datasets.
    sampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=max(len(dataset), batch_size)
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    while True:
        yield from loader


def train(dataset, output, *, recovery=None, checkpoint=None, steps=50000, batch_size=16,
          device="cpu", seed=0):
    if min(steps, batch_size) < 1 or (recovery is not None and batch_size < 2):
        raise ValueError("Training requires positive counts and a batch size of at least two for co-training")
    torch.manual_seed(seed)
    # The upstream ACT program uses absolute actions at 10 Hz, starting 80 ms ahead.
    timestamps = {"action": [0.08 + i / 10 for i in range(16)]}

    def open_dataset(path):
        return LeRobotDataset(repo_id=Path(path).name, root=path, delta_timestamps=timestamps,
                              video_backend="pyav")

    base = open_dataset(dataset)
    cameras = base.meta.camera_keys
    features = dataset_to_policy_features(base.meta.features)
    if not cameras or features["action"].shape != (10,):
        raise ValueError("Expected RGB cameras and 10-dimensional absolute robot actions")
    inputs = {key: PolicyFeature(type=features[key].type, shape=(3, *RGB_IMAGE_SIZE)) for key in cameras}
    inputs["observation.state"] = features["observation.state"]
    keys = [*inputs, "action", "action_is_pad"]
    recovery_count = batch_size // 2 if recovery is not None else 0
    batches = [training_batches(base, batch_size - recovery_count)]
    if recovery is not None:
        corrections = open_dataset(recovery)
        if any(base.meta.features[k] != corrections.meta.features.get(k) for k in [*inputs, "action"]):
            raise ValueError("Base and recovery camera, state, and action features must match")
        batches.append(training_batches(corrections, recovery_count))

    if checkpoint:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, ACTConfig):
            raise ValueError("Fine-tuning requires an ACT checkpoint")
        if (config.input_features != inputs or config.output_features != {"action": features["action"]}
                or config.chunk_size != len(timestamps["action"])):
            raise ValueError("Checkpoint camera, state, or action features differ from the dataset")
        config.device = device
        policy = ACTPolicy.from_pretrained(checkpoint, config=config)
    else:
        config = ACTConfig(input_features=inputs, output_features={"action": features["action"]},
                           chunk_size=16, n_action_steps=1, device=device,
                           normalization_mapping={k: NormalizationMode.MIN_MAX for k in ("VISUAL", "STATE", "ACTION")})
        policy = ACTPolicy(config)
    stats = deepcopy(base.meta.stats)
    # Use stock LeRobot MIN_MAX with identity bounds for rotation and gripper channels.
    stats["action"]["min"][3:] = -1
    stats["action"]["max"][3:] = 1
    pre, post = make_or_load_act_processors(config, stats, checkpoint)
    policy.to(device).train()
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for step in range(1, steps + 1):
        parts = [next(stream) for stream in batches]
        batch = {key: torch.cat([part[key] for part in parts]) for key in keys}
        batch = pre(resize_rgb_observation(batch, cameras))
        batch["action"][..., :3].clamp_(-1.2, 1.2)
        loss, _ = policy(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"step {step}/{steps}  loss {loss.item():.4f}", flush=True)
        if step % 5000 == 0 and step < steps:
            save_act_checkpoint(policy, pre, post, output / f"step_{step}")
    save_act_checkpoint(policy, pre, post, output / "final")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, help="Mix the recovery dataset into each training batch")
    parser.add_argument("--checkpoint", type=Path, help="Fine-tune a saved ACT policy and its processors")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

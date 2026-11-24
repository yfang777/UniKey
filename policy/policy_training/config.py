"""Policy configuration loading shared by training, evaluation, and collection."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml


SUPPORTED_POLICIES = ("ACT", "diffusion_encoder_v2", "Pi05")


@dataclass(frozen=True)
class PolicyRuntime:
    policy_key: str
    config: dict
    chunk_size: int
    policy_freq: int
    lerobot_fps: int
    use_relative_action: bool


def load_policy_config(path, policy, preset="default"):
    """Resolve a policy and optional section overrides into an independent value."""
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported policy: {policy}")
    with Path(path).open() as stream:
        document = yaml.safe_load(stream)
    config = deepcopy(document[policy])
    if preset == "default":
        return config
    overrides = document["presets"][preset]
    return {
        **config,
        **{
            section: {**config[section], **deepcopy(values)}
            for section, values in overrides.items()
        },
    }


def read_policy_runtime(policy, preset="default"):
    path = Path(f"configurations/policy/{policy}.yaml")
    config = load_policy_config(path, policy, preset)
    setting = config["setting"]
    chunk_size = int(setting["chunk_size"])
    policy_freq = int(setting["policy_infer_freq"])
    lerobot_fps = int(setting["lerobot_fps"])
    assert chunk_size == 16, f"Expected chunk_size=16 in {path}, got {chunk_size}"
    assert policy_freq == 10, f"Expected policy_infer_freq=10 in {path}, got {policy_freq}"
    assert lerobot_fps == 50, f"Expected lerobot_fps=50 in {path}, got {lerobot_fps}"
    return PolicyRuntime(
        policy_key=policy,
        config={policy: config},
        chunk_size=chunk_size,
        policy_freq=policy_freq,
        lerobot_fps=lerobot_fps,
        use_relative_action=bool(setting["use_relative_action"]),
    )

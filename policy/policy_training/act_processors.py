from copy import deepcopy
from pathlib import Path

from lerobot.configs.types import NormalizationMode
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

import torch
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def configure_absolute_actions(config, stats):
    config.normalization_mapping.update(VISUAL=NormalizationMode.MIN_MAX, STATE=NormalizationMode.MIN_MAX)
    if hasattr(NormalizationMode, "ABSOLUTE_ACTION_MIN_MAX_V1"):
        config.normalization_mapping["ACTION"] = NormalizationMode.ABSOLUTE_ACTION_MIN_MAX_V1
        return stats
    # Stock LeRobot can express identity rotation and gripper channels with these bounds.
    config.normalization_mapping["ACTION"] = NormalizationMode.MIN_MAX
    stats = deepcopy(stats)
    stats["action"]["min"][3:] = -1
    stats["action"]["max"][3:] = 1
    return stats


def load_act_for_rollout(ckpt_path, device):
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.device = device
    policy = ACTPolicy.from_pretrained(ckpt_path, config=config).to(device)
    processors = make_or_load_act_processors(config, None, ckpt_path)
    return ActActionWrapper(policy, *processors)


def make_or_load_act_processors(config, dataset_stats, pretrained_ckpt_path):
    if pretrained_ckpt_path is None:
        return make_act_pre_post_processors(config, dataset_stats)

    ckpt_path = Path(pretrained_ckpt_path)
    preprocessor_path = ckpt_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    postprocessor_path = ckpt_path / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    if not preprocessor_path.exists() or not postprocessor_path.exists():
        raise FileNotFoundError(
            f"ACT checkpoint {ckpt_path} is missing saved processor files. "
            "Train the first round with ACT processors before finetuning."
        )

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        ckpt_path,
        config_filename=preprocessor_path.name,
        overrides={"device_processor": {"device": config.device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        ckpt_path,
        config_filename=postprocessor_path.name,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def save_act_checkpoint(policy, preprocessor, postprocessor, ckpt_path):
    policy.save_pretrained(ckpt_path)
    preprocessor.save_pretrained(ckpt_path)
    postprocessor.save_pretrained(ckpt_path)


class ActActionWrapper:
    def __init__(self, policy, preprocessor, postprocessor):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def reset(self):
        self.policy.reset()

    def eval(self):
        self.policy.eval()
        return self

    def to(self, device):
        self.policy.to(device)
        return self

    @torch.no_grad()
    def predict_action_chunk(self, observation):
        batch = self.preprocessor(dict(observation))
        normalized_action = self.policy.predict_action_chunk(batch)
        action = self.postprocessor(normalized_action)
        action[..., 9].clamp_(0.0, 0.85)
        return action

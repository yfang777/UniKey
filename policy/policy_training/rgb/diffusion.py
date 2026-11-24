from pathlib import Path

import torch
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_or_load_diffusion_processors(config, dataset_stats, pretrained_ckpt_path):
    if pretrained_ckpt_path is None:
        return make_diffusion_pre_post_processors(config, dataset_stats)

    ckpt_path = Path(pretrained_ckpt_path)
    preprocessor_path = ckpt_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    postprocessor_path = ckpt_path / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    if not preprocessor_path.exists() or not postprocessor_path.exists():
        raise FileNotFoundError(
            f"Diffusion checkpoint {ckpt_path} is missing saved processor files. "
            "Train the first round with diffusion processors before finetuning."
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


def save_diffusion_checkpoint(policy, preprocessor, postprocessor, ckpt_path):
    policy.save_pretrained(ckpt_path)
    preprocessor.save_pretrained(ckpt_path)
    postprocessor.save_pretrained(ckpt_path)


class DiffusionActionWrapper:
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
        del batch[ACTION]
        normalized_action = self.policy.predict_action_chunk(batch)
        return self.postprocessor(normalized_action)

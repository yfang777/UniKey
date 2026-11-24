from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import os
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING
from types import SimpleNamespace

if TYPE_CHECKING:
    import jax
    from openpi.models import model as _model, pi0_config
    from openpi.policies import policy as _policy
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

CODEGEN_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPENPI_ROOT = Path(
    os.environ.get("OPENPI_ROOT", CODEGEN_ROOT / "third_party" / "Codegen_Openpi")
).resolve()
DEFAULT_LEROBOT_ROOT = Path(
    os.environ.get("LEROBOT_ROOT", CODEGEN_ROOT / "third_party" / "Codegen_lerobot_v0.5.1")
).resolve()


import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def install_lerobot_common_compat() -> None:
    try:
        import lerobot.common.datasets.lerobot_dataset  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import types
    import lerobot.datasets.lerobot_dataset as legacy_lerobot_dataset

    common_module = sys.modules.setdefault("lerobot.common", types.ModuleType("lerobot.common"))
    datasets_module = sys.modules.setdefault("lerobot.common.datasets", types.ModuleType("lerobot.common.datasets"))
    sys.modules["lerobot.common.datasets.lerobot_dataset"] = legacy_lerobot_dataset
    setattr(datasets_module, "lerobot_dataset", legacy_lerobot_dataset)
    setattr(common_module, "datasets", datasets_module)


from policy.policy_training.rgb.dataset import RgbLeRobotDataset
from policy.policy_training.rgb.dataset_combined_v4 import (
    CombinedRgbLeRobotDatasetV4,
    RatioBatchSampler,
)
from policy.policy_training.utils import inject_noise
from keypoint_program_solver.ik import KinHelper
from keypoint_program_solver.utils import resolve_robot_paths
from typing_extensions import override


OPENPI_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")
OPENPI_PRETRAINED_CKPT = "gs://openpi-assets/checkpoints/pi05_base/params"
STATE_DIM = 10
ACTION_DIM = 10
MINMAX_EPS = 1e-6
NORMALIZATION_IDENTITY = "IDENTITY"
NORMALIZATION_MIN_MAX = "MIN_MAX"
NORMALIZATION_ABSOLUTE_STATE_V1 = "ABSOLUTE_STATE_V1"
NORMALIZATION_ABSOLUTE_ACTION_V1 = "ABSOLUTE_ACTION_V1"


@functools.cache
def import_jax_modules():
    import jax
    import jax.numpy as jnp

    return jax, jnp


@functools.cache
def import_openpi_modules() -> SimpleNamespace:
    install_lerobot_common_compat()

    import openpi.models.model as model
    import openpi.models.pi0_config as pi0_config
    import openpi.policies.policy as policy
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.shared.normalize as normalize
    import openpi.training.checkpoints as checkpoints
    import openpi.training.config as config
    import openpi.transforms as transforms

    return SimpleNamespace(
        model=model,
        pi0_config=pi0_config,
        policy=policy,
        nnx_utils=nnx_utils,
        normalize=normalize,
        checkpoints=checkpoints,
        config=config,
        transforms=transforms,
    )


def task_camera_keys(task_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(f"observation.images.{camera_name}" for camera_name in task_cfg["camera_names"])


def normalize_openpi_params_path(path: str | Path) -> str:
    path_str = str(path)
    if path_str.startswith("gs://"):
        return path_str if path_str.rstrip("/").endswith("/params") else f"{path_str.rstrip('/')}/params"

    local = Path(path_str).expanduser()
    if local.is_dir() and (local / "params").exists():
        return str(local / "params")
    return str(local)


def checkpoint_layout(ckpt_save_dir: str | Path) -> tuple[str, str, str]:
    ckpt_path = Path(ckpt_save_dir).expanduser().resolve()
    return str(ckpt_path.parent.parent), ckpt_path.parent.name, ckpt_path.name


def latest_step_dir(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    steps = sorted(
        (int(path.name), path)
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not steps:
        raise FileNotFoundError(f"No numeric OpenPI checkpoint steps found in {checkpoint_dir}")
    return steps[-1][1]


def update_final_checkpoint_link(checkpoint_dir: str | Path, step_dir: str | Path | None = None) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    step_dir = Path(step_dir) if step_dir is not None else latest_step_dir(checkpoint_dir)
    final_path = checkpoint_dir / "final"

    if final_path.is_symlink() or final_path.exists():
        if final_path.is_dir() and not final_path.is_symlink():
            raise FileExistsError(
                f"{final_path} exists as a real directory. Remove it before replacing it with a symlink."
            )
        final_path.unlink()

    final_path.symlink_to(step_dir.name, target_is_directory=True)
    return final_path


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _feature_stats_to_minmax_norm_stats(stats: Mapping[str, Any], feature_key: str) -> _normalize.NormStats:
    missing = [name for name in ("mean", "std", "min", "max") if name not in stats]
    if missing:
        raise ValueError(f"LeRobot stats for {feature_key!r} are missing required min-max keys: {missing}")
    openpi = import_openpi_modules()
    return openpi.normalize.NormStats(
        mean=_to_numpy(stats["mean"]),
        std=_to_numpy(stats["std"]),
        q01=_to_numpy(stats["min"]),
        q99=_to_numpy(stats["max"]),
    )


def lerobot_stats_to_openpi_minmax_stats(
    dataset_stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, _normalize.NormStats]:
    required = {
        "state": "observation.state",
        "actions": "action",
    }
    missing = [source for source in required.values() if source not in dataset_stats]
    if missing:
        raise ValueError(f"LeRobot dataset stats are missing required keys: {missing}")
    return {
        target: _feature_stats_to_minmax_norm_stats(dataset_stats[source], source)
        for target, source in required.items()
    }


def load_minmax_stats_from_lerobot(repo_id: str, dataset_dir: str | Path) -> dict[str, _normalize.NormStats]:
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_dir)
    return lerobot_stats_to_openpi_minmax_stats(metadata.stats)


def load_minmax_stats_from_checkpoint(
    checkpoint_dir: str | Path,
    asset_id: str,
) -> dict[str, _normalize.NormStats]:
    openpi = import_openpi_modules()
    return openpi.checkpoints.load_norm_stats(Path(checkpoint_dir) / "assets", asset_id)


def _stats_bounds(stats: _normalize.NormStats, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Min-max normalization expects stats.q01=min and stats.q99=max.")
    low = np.asarray(stats.q01, dtype=np.float32)[..., :dim]
    high = np.asarray(stats.q99, dtype=np.float32)[..., :dim]
    return low, high


def _normalization_mode(value: str, *, name: str) -> str:
    normalized = str(value).strip().upper()
    if normalized == "ABSOLUTE_ACTION_MIN_MAX_V1":
        return NORMALIZATION_ABSOLUTE_ACTION_V1
    if normalized == "ABSOLUTE_STATE_MIN_MAX_V1":
        return NORMALIZATION_ABSOLUTE_STATE_V1
    if normalized in {
        NORMALIZATION_IDENTITY,
        NORMALIZATION_MIN_MAX,
        NORMALIZATION_ABSOLUTE_STATE_V1,
        NORMALIZATION_ABSOLUTE_ACTION_V1,
    }:
        return normalized
    raise ValueError(f"Unsupported {name}: {value!r}")


def minmax_normalize(value: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    low, high = _stats_bounds(stats, value.shape[-1])
    return (value - low) / (high - low + MINMAX_EPS) * 2.0 - 1.0


def minmax_unnormalize(value: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    low, high = _stats_bounds(stats, min(value.shape[-1], np.asarray(stats.q01).shape[-1]))
    dim = low.shape[-1]
    restored = (value[..., :dim] + 1.0) / 2.0 * (high - low + MINMAX_EPS) + low
    if dim < value.shape[-1]:
        restored = np.concatenate([restored, value[..., dim:]], axis=-1)
    return restored


def absolute_state_v1_normalize(value: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != STATE_DIM:
        raise ValueError(f"absolute_state_v1 expects state dimension {STATE_DIM}, got {value.shape[-1]}")
    low, high = _stats_bounds(stats, 3)
    output = value.copy()
    output[..., :3] = (output[..., :3] - low) / (high - low + MINMAX_EPS) * 2.0 - 1.0
    return output


def absolute_state_v1_unnormalize(value: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != STATE_DIM:
        raise ValueError(f"absolute_state_v1 expects state dimension {STATE_DIM}, got {value.shape[-1]}")
    low, high = _stats_bounds(stats, 3)
    output = value.copy()
    output[..., :3] = (output[..., :3] + 1.0) / 2.0 * (high - low + MINMAX_EPS) + low
    return output


def absolute_action_v1_normalize(value: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != ACTION_DIM:
        raise ValueError(f"absolute_action_v1 expects action dimension {ACTION_DIM}, got {value.shape[-1]}")
    low, high = _stats_bounds(stats, 3)
    output = value.copy()
    output[..., :3] = (output[..., :3] - low) / (high - low + MINMAX_EPS) * 2.0 - 1.0
    return output


def absolute_action_v1_unnormalize(value: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != ACTION_DIM:
        raise ValueError(f"absolute_action_v1 expects action dimension {ACTION_DIM}, got {value.shape[-1]}")
    low, high = _stats_bounds(stats, 3)
    output = value.copy()
    output[..., :3] = (output[..., :3] + 1.0) / 2.0 * (high - low + MINMAX_EPS) + low
    return output


def normalize_value(value: np.ndarray, stats: _normalize.NormStats, mode: str) -> np.ndarray:
    if mode == NORMALIZATION_IDENTITY:
        return np.asarray(value, dtype=np.float32)
    if mode == NORMALIZATION_MIN_MAX:
        return minmax_normalize(value, stats)
    if mode == NORMALIZATION_ABSOLUTE_STATE_V1:
        return absolute_state_v1_normalize(value, stats)
    if mode == NORMALIZATION_ABSOLUTE_ACTION_V1:
        return absolute_action_v1_normalize(value, stats)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def unnormalize_value(value: np.ndarray, stats: _normalize.NormStats, mode: str) -> np.ndarray:
    if mode == NORMALIZATION_IDENTITY:
        return np.asarray(value, dtype=np.float32)
    if mode == NORMALIZATION_MIN_MAX:
        return minmax_unnormalize(value, stats)
    if mode == NORMALIZATION_ABSOLUTE_STATE_V1:
        return absolute_state_v1_unnormalize(value, stats)
    if mode == NORMALIZATION_ABSOLUTE_ACTION_V1:
        return absolute_action_v1_unnormalize(value, stats)
    raise ValueError(f"Unsupported normalization mode: {mode}")


@dataclasses.dataclass(frozen=True)
class OpenPINormalize:
    norm_stats: Mapping[str, _normalize.NormStats]
    state_mode: str
    action_mode: str

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        if "state" in output:
            output["state"] = normalize_value(output["state"], self.norm_stats["state"], self.state_mode)
        if "actions" in output:
            output["actions"] = normalize_value(output["actions"], self.norm_stats["actions"], self.action_mode)
        return output


@dataclasses.dataclass(frozen=True)
class OpenPIUnnormalize:
    norm_stats: Mapping[str, _normalize.NormStats]
    action_mode: str

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        if "actions" in output:
            output["actions"] = unnormalize_value(output["actions"], self.norm_stats["actions"], self.action_mode)
        return output


def _parse_prompt(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _parse_prompt(value.item())
        return _parse_prompt(value.tolist())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected a single prompt value, got {value!r}")
        return _parse_prompt(value[0])
    return str(value)


def _parse_image(image: Any) -> np.ndarray:
    image = np.asarray(image.detach().cpu() if isinstance(image, torch.Tensor) else image)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"OpenPI pi0.5 expects one observation image per camera, got image shape {image.shape}")
        image = image[0]
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = (255.0 * image).clip(0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3] or [3, H, W], got {image.shape}")
    return image


def _slice_batch_value(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, np.ndarray):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def infer_batch_size(batch: Mapping[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
        if isinstance(value, np.ndarray):
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Unable to infer batch size from batch")


@dataclasses.dataclass(frozen=True)
class CodegenOpenPIInputs:
    camera_keys: tuple[str, ...]
    image_keys: tuple[str, ...] = OPENPI_IMAGE_KEYS
    no_ee: bool = False
    default_task: str | None = None
    require_actions: bool = False

    def __post_init__(self) -> None:
        if len(self.camera_keys) != len(self.image_keys):
            raise ValueError(f"Expected {len(self.image_keys)} LeRobot cameras, got {self.camera_keys}")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = _to_numpy(data["observation.state"])
        if state.ndim == 2:
            state = state[0]
        if self.no_ee:
            state = np.zeros_like(state)

        if self.default_task is None:
            raise ValueError("Cannot build OpenPI prompt without default_task")
        prompt = self.default_task

        output = {
            "image": {
                image_key: _parse_image(data[camera_key])
                for camera_key, image_key in zip(self.camera_keys, self.image_keys, strict=True)
            },
            "image_mask": {image_key: np.True_ for image_key in self.image_keys},
            "state": state,
            "prompt": prompt,
        }

        if "action" in data:
            actions = _to_numpy(data["action"])
            if actions.ndim != 2:
                raise ValueError(f"Expected action chunk with shape [T, D], got {actions.shape}")
            output["actions"] = actions
            output["action_is_pad"] = np.asarray(
                data.get("action_is_pad", np.zeros(actions.shape[0], dtype=np.bool_))
            ).astype(bool)
        elif self.require_actions:
            raise ValueError("Training input is missing required 'action'.")

        return output


@dataclasses.dataclass(frozen=True)
class CodegenOpenPIOutputs:
    action_dim: int = ACTION_DIM

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data:
            return data
        return {**data, "actions": np.asarray(data["actions"], dtype=np.float32)[..., : self.action_dim]}


def build_pi05_model_config(setting: Mapping[str, Any], image_keys: Sequence[str] = OPENPI_IMAGE_KEYS) -> pi0_config.Pi0Config:
    openpi = import_openpi_modules()
    state_discretization = setting.get("state_discretization")
    if state_discretization is None:
        state_discretization_min = -1.0
        state_discretization_max = 1.0
        state_discretization_bins = 256
    else:
        state_discretization_min = float(state_discretization["min_value"])
        state_discretization_max = float(state_discretization["max_value"])
        state_discretization_bins = int(state_discretization["num_bins"])
    return openpi.pi0_config.Pi0Config(
        pi05=True,
        dtype=str(setting.get("dtype", "bfloat16")),
        paligemma_variant=setting.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=setting.get("action_expert_variant", "gemma_300m"),
        action_dim=int(setting.get("max_action_dim", 32)),
        action_horizon=int(setting["chunk_size"]),
        max_token_len=int(setting.get("tokenizer_max_length", 200)),
        image_keys=tuple(image_keys),
        discrete_state_input=bool(setting.get("discrete_state_input", True)),
        continuous_state_input=bool(setting.get("continuous_state_input", False)),
        state_discretization_min=state_discretization_min,
        state_discretization_max=state_discretization_max,
        state_discretization_bins=state_discretization_bins,
    )


def build_data_config(
    *,
    repo_id: str,
    norm_stats: dict[str, _normalize.NormStats],
    model_config: _model.BaseModelConfig,
    camera_keys: tuple[str, ...],
    image_keys: tuple[str, ...],
    image_normalization: str,
    state_normalization: str,
    action_normalization: str,
    no_ee: bool = False,
    default_task: str | None = None,
    require_actions: bool = False,
) -> _config.DataConfig:
    openpi = import_openpi_modules()
    model_transforms = openpi.config.ModelTransformFactory()(model_config)
    image_mode = _normalization_mode(image_normalization, name="image_normalization")
    if image_mode != NORMALIZATION_IDENTITY:
        raise ValueError("Pi05 expects image_normalization=IDENTITY.")
    state_mode = _normalization_mode(state_normalization, name="state_normalization")
    action_mode = _normalization_mode(action_normalization, name="action_normalization")
    return openpi.config.DataConfig(
        repo_id=repo_id,
        asset_id=repo_id,
        norm_stats=norm_stats,
        data_transforms=openpi.transforms.Group(
            inputs=[
                CodegenOpenPIInputs(
                    camera_keys=camera_keys,
                    image_keys=image_keys,
                    no_ee=no_ee,
                    default_task=default_task,
                    require_actions=require_actions,
                ),
                OpenPINormalize(
                    norm_stats,
                    state_mode=state_mode,
                    action_mode=action_mode,
                ),
                openpi.transforms.PadStatesAndActions(model_config.action_dim),
            ],
            outputs=[
                CodegenOpenPIOutputs(action_dim=ACTION_DIM),
                OpenPIUnnormalize(norm_stats, action_mode=action_mode),
            ],
        ),
        model_transforms=model_transforms,
        use_quantile_norm=True,
        action_sequence_keys=("action",),
        prompt_from_task=False,
    )


@dataclasses.dataclass(frozen=True)
class CodegenOpenPIDataConfigFactory:
    repo_id: str
    action_timestamp_offset: float
    robot_variant: str
    dataset_dir: str = dataclasses.field(default="")
    recovery_dataset_path: str | None = None
    new_recovery_episode_count: int = 0
    dataset_ratios: tuple[tuple[str, float], ...] = dataclasses.field(default_factory=tuple)
    stats_checkpoint_dir: str | None = None
    camera_keys: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    image_keys: tuple[str, ...] = OPENPI_IMAGE_KEYS
    image_normalization: str = NORMALIZATION_IDENTITY
    state_normalization: str = NORMALIZATION_MIN_MAX
    action_normalization: str = NORMALIZATION_MIN_MAX
    policy_infer_freq: int = 10
    n_obs_steps: int = 1
    chunk_size: int = 16
    no_ee: bool = False
    default_task: str | None = None

    def __post_init__(self) -> None:
        if self.n_obs_steps != 1:
            raise ValueError("OpenPI pi0.5 training requires n_obs_steps == 1.")
        if len(self.camera_keys) != len(self.image_keys):
            raise ValueError(f"Expected {len(self.image_keys)} cameras for image keys {self.image_keys}, got {self.camera_keys}")

    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        del assets_dirs
        if self.stats_checkpoint_dir is None:
            norm_stats = load_minmax_stats_from_lerobot(self.repo_id, self.dataset_dir)
        else:
            norm_stats = load_minmax_stats_from_checkpoint(self.stats_checkpoint_dir, self.repo_id)
        return build_data_config(
            repo_id=self.repo_id,
            norm_stats=norm_stats,
            model_config=model_config,
            camera_keys=self.camera_keys,
            image_keys=self.image_keys,
            image_normalization=self.image_normalization,
            state_normalization=self.state_normalization,
            action_normalization=self.action_normalization,
            no_ee=self.no_ee,
            default_task=self.default_task,
            require_actions=True,
        )

    def delta_timestamps(self) -> dict[str, list[float]]:
        delta_timestamps = {
            "observation.state": [i / self.policy_infer_freq for i in range(1 - self.n_obs_steps, 1)],
            "action": [i / self.policy_infer_freq + self.action_timestamp_offset for i in range(self.chunk_size)],
        }
        for camera_key in self.camera_keys:
            delta_timestamps[camera_key] = [i / self.policy_infer_freq for i in range(1 - self.n_obs_steps, 1)]
        return delta_timestamps

    def create_dataset(self) -> RgbLeRobotDataset | CombinedRgbLeRobotDatasetV4:
        required_columns = [
            *self.camera_keys,
            "observation.state",
            "action",
            "episode_index",
            "task_index",
        ]
        if self.recovery_dataset_path is None:
            return RgbLeRobotDataset(
                required_columns=required_columns,
                repo_id=self.repo_id,
                root=self.dataset_dir,
                delta_timestamps=self.delta_timestamps(),
                video_backend=None,
                image_transforms=None,
            )
        return CombinedRgbLeRobotDatasetV4(
            required_columns=required_columns,
            recovery_dataset_path=self.recovery_dataset_path,
            new_recovery_episode_count=self.new_recovery_episode_count,
            repo_id=self.repo_id,
            root=self.dataset_dir,
            delta_timestamps=self.delta_timestamps(),
            video_backend=None,
            image_transforms=None,
        )

    def create_batch_sampler(
        self,
        dataset: RgbLeRobotDataset | CombinedRgbLeRobotDatasetV4,
        batch_size: int,
        seed: int,
    ) -> RatioBatchSampler | None:
        if self.recovery_dataset_path is None:
            return None
        return RatioBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            ratios=dict(self.dataset_ratios),
            seed=seed,
        )


class CodegenOpenPIDataLoader:
    def __init__(
        self,
        dataset: RgbLeRobotDataset,
        data_config: _config.DataConfig,
        *,
        batch_size: int,
        sharding_spec: jax.sharding.Sharding | None,
        shuffle: bool,
        num_batches: int | None,
        num_workers: int,
        seed: int,
        add_action_noise: bool,
        urdf_path: Path,
        batch_sampler: torch.utils.data.Sampler | None,
    ):
        jax, _ = import_jax_modules()
        openpi = import_openpi_modules()
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple JAX processes is not supported.")
        if batch_size % jax.process_count() != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by JAX process count {jax.process_count()}.")

        self._data_config = data_config
        self._transform = openpi.transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                *data_config.model_transforms.inputs,
            ]
        )
        self._sharding = sharding_spec
        self._num_batches = num_batches
        self._kin_helper = KinHelper(urdf_path) if add_action_noise else None

        local_batch_size = batch_size // jax.process_count()
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than dataset size ({len(dataset)}).")

        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {
            "num_workers": num_workers,
            "multiprocessing_context": "spawn" if num_workers > 0 else None,
            "persistent_workers": num_workers > 0,
            "pin_memory": False,
            "generator": generator,
        }
        if batch_sampler is None:
            loader_kwargs.update({
                "batch_size": local_batch_size,
                "shuffle": shuffle,
                "drop_last": True,
            })
        else:
            loader_kwargs["batch_sampler"] = batch_sampler
        self._torch_loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        jax, _ = import_jax_modules()
        openpi = import_openpi_modules()
        num_items = 0
        while True:
            for raw_batch in self._torch_loader:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                num_items += 1
                if self._kin_helper is not None:
                    raw_batch = inject_noise(self._kin_helper, raw_batch)
                batch = self._transform_batch(raw_batch)
                if self._sharding is not None:
                    batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                action_is_pad = batch.pop("action_is_pad")
                yield openpi.model.Observation.from_dict(batch), batch["actions"], action_is_pad

    def _transform_batch(self, raw_batch: Mapping[str, Any]) -> dict[str, Any]:
        jax, _ = import_jax_modules()
        batch_size = infer_batch_size(raw_batch)
        items = []
        for index in range(batch_size):
            item = {key: _slice_batch_value(value, index) for key, value in raw_batch.items()}
            items.append(self._transform(item))
        return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def create_data_loader(
    train_config: _config.TrainConfig,
    data_factory: CodegenOpenPIDataConfigFactory,
    *,
    sharding_spec: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    add_action_noise: bool = False,
) -> CodegenOpenPIDataLoader:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = data_factory.create_dataset()
    batch_sampler = data_factory.create_batch_sampler(
        dataset,
        train_config.batch_size,
        train_config.seed,
    )
    return CodegenOpenPIDataLoader(
        dataset,
        data_config,
        batch_size=train_config.batch_size,
        sharding_spec=sharding_spec,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
        add_action_noise=add_action_noise,
        urdf_path=resolve_robot_paths(data_factory.robot_variant)[0],
        batch_sampler=batch_sampler,
    )


class OpenPIActionWrapper:
    def __init__(
        self,
        *,
        model_config: _model.BaseModelConfig,
        checkpoint_dir: str | Path,
        norm_stats: dict[str, _normalize.NormStats],
        repo_id: str,
        camera_keys: tuple[str, ...],
        image_keys: tuple[str, ...],
        image_normalization: str,
        state_normalization: str,
        action_normalization: str,
        default_task: str,
        device: str,
        seed: int = 0,
        num_inference_steps: int = 10,
    ):
        self.device = device
        self.default_task = default_task
        self.num_inference_steps = num_inference_steps

        jax, jnp = import_jax_modules()
        openpi = import_openpi_modules()
        checkpoint_dir = Path(checkpoint_dir).expanduser()
        params = openpi.model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        self._model = model_config.load(params)
        self._sample_actions = openpi.nnx_utils.module_jit(self._model.sample_actions)
        self._rng = jax.random.key(seed)

        data_config = build_data_config(
            repo_id=repo_id,
            norm_stats=norm_stats,
            model_config=model_config,
            camera_keys=camera_keys,
            image_keys=image_keys,
            image_normalization=image_normalization,
            state_normalization=state_normalization,
            action_normalization=action_normalization,
            default_task=default_task,
            require_actions=False,
        )
        self._input_transform = openpi.transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = openpi.transforms.compose(
            [
                *data_config.model_transforms.outputs,
                *data_config.data_transforms.outputs,
                *data_config.repack_transforms.outputs,
            ]
        )

    def reset(self) -> None:
        return None

    def eval(self) -> "OpenPIActionWrapper":
        return self

    def to(self, device: str) -> "OpenPIActionWrapper":
        self.device = device
        return self

    @torch.no_grad()
    def predict_action_chunk(self, observation: Mapping[str, Any]) -> torch.Tensor:
        jax, jnp = import_jax_modules()
        openpi = import_openpi_modules()
        batch = self._transform_observation_batch(observation)
        inputs = jax.tree.map(jnp.asarray, batch)
        obs = openpi.model.Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, obs, num_steps=self.num_inference_steps),
        }
        outputs = jax.tree.map(np.asarray, outputs)
        outputs = self._output_transform(outputs)
        return torch.from_numpy(outputs["actions"]).to(self.device)

    def _transform_observation_batch(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        jax, _ = import_jax_modules()
        batch_size = infer_batch_size(observation)
        items = []
        for index in range(batch_size):
            item = {key: _slice_batch_value(value, index) for key, value in observation.items()}
            items.append(self._input_transform(item))
        return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def build_openpi_eval_wrapper(
    *,
    setting_cfg: Mapping[str, Any],
    checkpoint_dir: str | Path,
    repo_id: str,
    camera_keys: tuple[str, ...],
    image_keys: tuple[str, ...],
    task_description: str,
    device: str,
    seed: int,
) -> _policy.BasePolicy:
    model_config = build_pi05_model_config(setting_cfg, image_keys)
    norm_stats = load_minmax_stats_from_checkpoint(checkpoint_dir, repo_id)
    return OpenPIActionWrapper(
        model_config=model_config,
        checkpoint_dir=checkpoint_dir,
        norm_stats=norm_stats,
        repo_id=repo_id,
        camera_keys=camera_keys,
        image_keys=image_keys,
        image_normalization=str(setting_cfg["image_normalization"]),
        state_normalization=str(setting_cfg["state_normalization"]),
        action_normalization=str(setting_cfg["action_normalization"]),
        default_task=task_description,
        device=device,
        seed=seed,
        num_inference_steps=int(setting_cfg.get("num_inference_steps", 10)),
    )

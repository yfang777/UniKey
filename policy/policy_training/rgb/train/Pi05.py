from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import functools
import logging
import os
from pathlib import Path
import platform
import random
import re
from typing import Any

import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import wandb
import yaml

from policy.policy_training.config import load_policy_config

try:
    import tqdm_loggable.auto as tqdm
except ImportError:  # pragma: no cover
    import tqdm

from policy.policy_training.rgb.openpi import (
    ACTION_DIM,
    DEFAULT_LEROBOT_ROOT,
    DEFAULT_OPENPI_ROOT,
    OPENPI_IMAGE_KEYS,
    OPENPI_PRETRAINED_CKPT,
    CodegenOpenPIDataConfigFactory,
    build_pi05_model_config,
    checkpoint_layout,
    create_data_loader,
    install_lerobot_common_compat,
    normalize_openpi_params_path,
    task_camera_keys,
    update_final_checkpoint_link,
)

install_lerobot_common_compat()

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


POLICY_KEY = "Pi05"
VISUAL_ENCODER_FILTER = nnx_utils.PathRegex(".*PaliGemma/img/.*")
ACTION_EXPERT_FILTER = nnx_utils.PathRegex(".*PaliGemma/llm/.*_1.*")
LLM_FILTER = nnx_utils.PathRegex(".*PaliGemma/llm/.*")
LORA_FILTER = nnx_utils.PathRegex(".*lora.*")
ACTION_EXPERT_TRAINABLE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r".*PaliGemma/llm/.*_1.*",
        r".*action_in_proj/.*",
        r".*action_out_proj/.*",
        r".*time_mlp_in/.*",
        r".*time_mlp_out/.*",
        r".*continuous_state_proj/.*",
    )
)


@dataclasses.dataclass(frozen=True)
class EMASchedule:
    update_after_step: int
    inv_gamma: float
    power: float
    min_value: float
    max_value: float


@dataclasses.dataclass(frozen=True)
class CodegenOpenPITrainConfig(_config.TrainConfig):
    ema_schedule: EMASchedule | None = None
    visual_encoder_lr_scale: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Codegen RGB policy with JAX OpenPI pi0.5.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--recovery_dataset_path", type=str, default=None)
    parser.add_argument("--new_recovery_episode_count", type=int, default=0)
    parser.add_argument("--policy_config_path", type=str, required=True)
    parser.add_argument("--policy_preset", default="default")
    parser.add_argument("--pretrained_ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None, help="Accepted for compatibility; JAX selects devices.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--tasks_config_path", type=str, default="configurations/tasks.yaml")
    parser.add_argument("--no_ee", action="store_true")
    parser.add_argument("--use_absolute_action", action="store_true")
    parser.add_argument("--add_action_noise", action="store_true")
    parser.add_argument(
        "--save_intermediate_ckpts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save periodic numeric OpenPI checkpoints in addition to the final symlink.",
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=None,
        help="Override the config log interval.",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fsdp_devices", "--fsdp-devices", dest="fsdp_devices", type=int, default=None)
    parser.add_argument("--num_train_steps", "--num-train-steps", dest="num_train_steps", type=int, default=None)
    parser.add_argument("--save_interval", "--save-interval", dest="save_interval", type=int, default=None)
    parser.add_argument("--batch_size_rgb", "--batch-size-rgb", dest="batch_size_rgb", type=int, default=None)
    parser.add_argument("--num_workers_rgb", "--num-workers-rgb", dest="num_workers_rgb", type=int, default=None)
    parser.add_argument(
        "--dry_run_batches",
        "--dry-run-batches",
        dest="dry_run_batches",
        type=int,
        default=0,
        help="Only build and print this many transformed batches; does not initialize model weights.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_logging() -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler())
    root_logger.handlers[0].setFormatter(formatter)
    logging.getLogger("orbax").setLevel(logging.WARNING)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True) -> None:
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            dir=os.environ.get("WANDB_DIR", os.path.expanduser("~/.cache/wandb")),
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def should_log_step(global_step: int, num_steps: int, log_every_steps: int) -> bool:
    return global_step == 1 or global_step % log_every_steps == 0 or global_step == num_steps


def _load_yaml(path: str | Path, key: str) -> dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if key not in config:
        raise KeyError(f"{path} does not contain top-level key {key!r}; available keys: {tuple(config)}")
    return config[key]




def _train_only_action_expert_filter(path: nnx.filterlib.PathParts, _: Any) -> bool:
    joined_path = "/".join(str(part) for part in path)
    return any(pattern.fullmatch(joined_path) is not None for pattern in ACTION_EXPERT_TRAINABLE_PATTERNS)


def _freeze_llm_except_action_expert(path: nnx.filterlib.PathParts, value: Any) -> bool:
    return LLM_FILTER(path, value) and not ACTION_EXPERT_FILTER(path, value)


def _freeze_llm_except_lora(path: nnx.filterlib.PathParts, value: Any) -> bool:
    return LLM_FILTER(path, value) and not LORA_FILTER(path, value)


def _freeze_filter(setting_cfg: Mapping[str, Any]) -> nnx.filterlib.Filter:
    if bool(setting_cfg.get("train_expert_only", False)):
        return nnx.All(nnx.Param, nnx.Not(_train_only_action_expert_filter))
    filters = []
    if bool(setting_cfg.get("freeze_encoder", False)) or bool(setting_cfg.get("freeze_vision_encoder", False)):
        filters.append(VISUAL_ENCODER_FILTER)
    if bool(setting_cfg.get("train_lora", False)):
        filters.append(_freeze_llm_except_lora)
    elif bool(setting_cfg.get("freeze_llm", False)):
        filters.append(_freeze_llm_except_action_expert)
    if not filters:
        return nnx.Nothing

    def freeze_any(path: nnx.filterlib.PathParts, value: Any) -> bool:
        return any(filter_fn(path, value) for filter_fn in filters)

    return freeze_any


def _param_count(params: nnx.State) -> int:
    return sum(int(np.prod(param.value.shape)) for param in params.flat_state().values())


def _state_paths(params: nnx.State, limit: int) -> list[str]:
    return ["/".join(str(part) for part in path) for path in list(params.flat_state())[:limit]]


def _scaled_schedule(schedule: optax.Schedule, scale: float) -> optax.Schedule:
    return lambda step: scale * schedule(step)


def _ema_decay_for_step(config: CodegenOpenPITrainConfig, optimization_step: at.Int[at.ArrayLike, ""]):
    if config.ema_schedule is None:
        return config.ema_decay

    schedule = config.ema_schedule
    step = jnp.maximum(0, optimization_step - schedule.update_after_step - 1)
    value = 1 - (1 + step / schedule.inv_gamma) ** -schedule.power
    return jnp.where(step <= 0, 0.0, jnp.clip(value, schedule.min_value, schedule.max_value))


def _optimizer_labels(
    params: nnx.State,
    visual_encoder_filter: nnx.filterlib.Filter,
) -> nnx.State:
    visual_encoder_keys = set(params.filter(visual_encoder_filter).flat_state())
    if not visual_encoder_keys:
        raise ValueError("visual_encoder_lr_scale was set, but no visual encoder parameters matched.")
    return params.map(
        lambda path, _: (
            "visual_encoder" if path in visual_encoder_keys else "policy"
        )
    )


def _create_train_optimizer(
    config: CodegenOpenPITrainConfig,
) -> optax.GradientTransformation:
    lr_schedule = config.lr_schedule.create()
    if config.visual_encoder_lr_scale is None:
        return config.optimizer.create(lr_schedule, weight_decay_mask=None)

    visual_encoder_lr_schedule = _scaled_schedule(
        lr_schedule,
        float(config.visual_encoder_lr_scale),
    )
    return optax.multi_transform(
        {
            "policy": config.optimizer.create(lr_schedule, weight_decay_mask=None),
            "visual_encoder": config.optimizer.create(
                visual_encoder_lr_schedule,
                weight_decay_mask=None,
            ),
        },
        lambda params: _optimizer_labels(params, VISUAL_ENCODER_FILTER),
    )


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: CodegenOpenPITrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _create_train_optimizer(config)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        trainable_params = params.filter(config.trainable_filter)
        use_ema = config.ema_decay is not None or config.ema_schedule is not None

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(trainable_params),
            ema_decay=config.ema_decay,
            ema_params=params if use_ema else None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: CodegenOpenPITrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, at.Bool[at.Array, "b ah"]],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        action_is_pad: at.Bool[at.Array, "b ah"],
    ):
        if hasattr(model, "compute_loss_per_dim"):
            per_dim_loss = model.compute_loss_per_dim(rng, observation, actions, train=True)
            valid_steps = jnp.logical_not(action_is_pad[:, : per_dim_loss.shape[1]])
            valid_dims = jnp.arange(per_dim_loss.shape[2]) < ACTION_DIM
            valid = jnp.logical_and(valid_steps[:, :, None], valid_dims[None, None, :])
            valid_count = jnp.maximum(jnp.sum(valid), 1)
            return jnp.sum(per_dim_loss * valid) / valid_count

        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        valid_steps = jnp.logical_not(action_is_pad[:, : chunked_loss.shape[1]])
        valid_count = jnp.maximum(jnp.sum(valid_steps), 1)
        return jnp.sum(chunked_loss * valid_steps) / valid_count

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, action_is_pad = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions, action_is_pad)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    ema_decay = _ema_decay_for_step(config, state.step)
    if ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: ema_decay * old + (1 - ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return new_state, {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "valid_action_steps": jnp.sum(jnp.logical_not(action_is_pad)),
        **({} if ema_decay is None else {"ema_decay": jnp.asarray(ema_decay)}),
    }


def build_train_config(args: argparse.Namespace) -> tuple[CodegenOpenPITrainConfig, CodegenOpenPIDataConfigFactory]:
    user_config = load_policy_config(args.policy_config_path, POLICY_KEY, args.policy_preset)
    setting_cfg = user_config["setting"]
    train_cfg = user_config["train"]

    if not args.use_absolute_action:
        raise ValueError("Pi05 training is wired for absolute actions; pass --use_absolute_action.")
    if bool(setting_cfg.get("use_relative_action", False)):
        raise ValueError("Pi05 expects setting.use_relative_action=False.")
    freeze_filter = _freeze_filter(setting_cfg)

    with open(args.tasks_config_path, "r") as f:
        tasks_config = yaml.safe_load(f)["tasks"]
    task_cfg = tasks_config[args.task]
    task_description = str(task_cfg["task_description"])
    camera_keys = task_camera_keys(task_cfg)
    image_keys = tuple(setting_cfg.get("image_keys", OPENPI_IMAGE_KEYS))

    pretrained_path = args.pretrained_ckpt_path
    stats_checkpoint_dir = None
    if pretrained_path is None:
        pretrained_path = setting_cfg.get("pretrained_policy_path", OPENPI_PRETRAINED_CKPT)
        num_train_steps = int(train_cfg["num_steps_rgb"])
        save_interval = int(train_cfg["save_freq"])
    else:
        params_path = Path(normalize_openpi_params_path(pretrained_path))
        stats_checkpoint_dir = str(params_path.parent if params_path.name == "params" else params_path)
        num_train_steps = int(train_cfg["num_steps_finetune"])
        save_interval = int(train_cfg["save_freq_finetune"])

    if args.num_train_steps is not None:
        num_train_steps = int(args.num_train_steps)
    if args.save_interval is not None:
        save_interval = int(args.save_interval)
    if num_train_steps <= 0:
        raise ValueError("--num_train_steps must be > 0")
    if save_interval <= 0:
        raise ValueError("--save_interval must be > 0")

    batch_size = int(train_cfg["batch_size_rgb"] if args.batch_size_rgb is None else args.batch_size_rgb)
    num_workers = int(train_cfg["num_workers_rgb"] if args.num_workers_rgb is None else args.num_workers_rgb)
    if batch_size <= 0:
        raise ValueError("--batch_size_rgb must be > 0")
    if num_workers < 0:
        raise ValueError("--num_workers_rgb must be >= 0")
    visual_encoder_lr_scale = setting_cfg.get("visual_encoder_lr_scale")
    if visual_encoder_lr_scale is not None and float(visual_encoder_lr_scale) <= 0:
        raise ValueError("visual_encoder_lr_scale must be > 0")

    checkpoint_base_dir, config_name, exp_name = checkpoint_layout(args.ckpt_save_dir)
    log_interval = int(args.log_every_steps or train_cfg.get("log_every_steps", 1000))
    if log_interval <= 0:
        raise ValueError("--log_every_steps must be > 0")

    data_factory = CodegenOpenPIDataConfigFactory(
        robot_variant=task_cfg["robot_variant"],
        repo_id=str(setting_cfg["repo_id"]),
        dataset_dir=args.dataset_dir,
        recovery_dataset_path=args.recovery_dataset_path,
        new_recovery_episode_count=int(args.new_recovery_episode_count),
        dataset_ratios=(
            ("offline", float(train_cfg["dataset_combined_v4"]["offline"])),
            ("new_recovery", float(train_cfg["dataset_combined_v4"]["new_recovery"])),
        ) if args.recovery_dataset_path is not None else (),
        stats_checkpoint_dir=stats_checkpoint_dir,
        camera_keys=camera_keys,
        image_keys=image_keys,
        image_normalization=str(setting_cfg["image_normalization"]),
        state_normalization=str(setting_cfg["state_normalization"]),
        action_normalization=str(setting_cfg["action_normalization"]),
        policy_infer_freq=int(setting_cfg["policy_infer_freq"]),
        n_obs_steps=int(setting_cfg["n_obs_steps"]),
        chunk_size=int(setting_cfg["chunk_size"]),
        action_timestamp_offset=float(setting_cfg["action_timestamp_offset"]),
        no_ee=args.no_ee,
        default_task=task_description,
    )

    optimizer_betas = tuple(setting_cfg.get("optimizer_betas", (0.9, 0.95)))
    if len(optimizer_betas) != 2:
        raise ValueError(f"optimizer_betas must have length 2, got {optimizer_betas}")

    ema_decay = train_cfg.get("ema_decay", 0.99)
    ema_schedule = None
    if "ema" in train_cfg:
        ema_cfg = train_cfg["ema"]
        if ema_cfg is False:
            ema_decay = None
        else:
            ema_decay = None
            ema_schedule = EMASchedule(
                update_after_step=int(ema_cfg["update_after_step"]),
                inv_gamma=float(ema_cfg["inv_gamma"]),
                power=float(ema_cfg["power"]),
                min_value=float(ema_cfg["min_value"]),
                max_value=float(ema_cfg["max_value"]),
            )

    model = build_pi05_model_config(setting_cfg, image_keys)
    fsdp_devices = int(args.fsdp_devices or train_cfg.get("fsdp_devices", 1))
    missing_param_regex = (
        ".*(lora|continuous_state_proj).*"
        if model.continuous_state_input
        else ".*lora.*"
    )

    return (
        CodegenOpenPITrainConfig(
            name=config_name,
            project_name=str(train_cfg.get("project_name", "Pi05_training")),
            exp_name=exp_name,
            model=model,
            weight_loader=_weight_loaders.CheckpointWeightLoader(
                normalize_openpi_params_path(pretrained_path),
                missing_regex=missing_param_regex,
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=int(setting_cfg.get("scheduler_warmup_steps", 1000)),
                peak_lr=float(setting_cfg.get("optimizer_lr", 2.5e-5)),
                decay_steps=int(setting_cfg.get("scheduler_decay_steps", num_train_steps)),
                decay_lr=float(setting_cfg.get("scheduler_decay_lr", 2.5e-6)),
            ),
            optimizer=_optimizer.AdamW(
                b1=float(optimizer_betas[0]),
                b2=float(optimizer_betas[1]),
                eps=float(setting_cfg.get("optimizer_eps", 1e-8)),
                weight_decay=float(setting_cfg.get("optimizer_weight_decay", 0.01)),
                clip_gradient_norm=float(setting_cfg.get("optimizer_grad_clip_norm", 1.0)),
            ),
            ema_decay=ema_decay,
            ema_schedule=ema_schedule,
            freeze_filter=freeze_filter,
            data=data_factory,
            checkpoint_base_dir=checkpoint_base_dir,
            seed=int(args.seed),
            batch_size=batch_size,
            num_workers=num_workers,
            num_train_steps=num_train_steps,
            log_interval=log_interval,
            save_interval=save_interval if args.save_intermediate_ckpts else num_train_steps,
            keep_period=train_cfg.get("keep_period", save_interval),
            overwrite=bool(args.overwrite),
            resume=bool(args.resume),
            wandb_enabled=bool(train_cfg.get("wandb_enabled", True) if args.wandb_enabled is None else args.wandb_enabled),
            fsdp_devices=fsdp_devices,
            visual_encoder_lr_scale=visual_encoder_lr_scale,
        ),
        data_factory,
    )


def dry_run(
    config: _config.TrainConfig,
    data_factory: CodegenOpenPIDataConfigFactory,
    num_batches: int,
    add_action_noise: bool,
) -> None:
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = create_data_loader(
        config,
        data_factory,
        sharding_spec=data_sharding,
        shuffle=False,
        num_batches=num_batches,
        add_action_noise=add_action_noise,
    )
    for index, batch in enumerate(loader, start=1):
        logging.info("Dry-run batch %d:\n%s", index, training_utils.array_tree_to_info(batch))


def train(config: CodegenOpenPITrainConfig, data_factory: CodegenOpenPIDataConfigFactory, *, add_action_noise: bool) -> None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path("~/.cache/jax").expanduser())),
    )
    logging.info("Running on: %s", platform.node())
    logging.info("JAX devices: %s", jax.devices())
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Global batch size {config.batch_size} must be divisible by JAX device count {jax.device_count()}.")

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = create_data_loader(
        config,
        data_factory,
        sharding_spec=data_sharding,
        shuffle=True,
        add_action_noise=add_action_noise,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info("Initialized data loader:\n%s", training_utils.array_tree_to_info(batch))

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))
    all_params = train_state.params.filter(nnx.Param)
    trainable_params = train_state.params.filter(config.trainable_filter)
    visual_encoder_params = trainable_params.filter(VISUAL_ENCODER_FILTER)
    llm_params = trainable_params.filter(LLM_FILTER)
    all_param_count = _param_count(all_params)
    trainable_param_count = _param_count(trainable_params)
    visual_encoder_param_count = _param_count(visual_encoder_params)
    llm_param_count = _param_count(llm_params)
    logging.info(
        "Trainable parameters: %d / %d (%.4f%%)",
        trainable_param_count,
        all_param_count,
        100.0 * trainable_param_count / all_param_count,
    )
    logging.info("Trainable visual encoder parameters: %d", visual_encoder_param_count)
    logging.info("Trainable LLM parameters: %d", llm_param_count)
    if config.visual_encoder_lr_scale is not None:
        logging.info(
            "Visual encoder learning rate scale: %.6f",
            float(config.visual_encoder_lr_scale),
        )
    logging.info("Example trainable parameter paths: %s", _state_paths(trainable_params, 20))

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        desc="OpenPI pi0.5 training",
    )
    infos = []
    latest_saved_step = start_step
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        global_step = step + 1
        if should_log_step(global_step, config.num_train_steps, config.log_interval):
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {global_step}: {info_str}")
            wandb.log(reduced_info, step=global_step)
            infos = []

        batch = next(data_iter)
        if (
            (config.save_interval > 0 and global_step % config.save_interval == 0)
            or global_step == config.num_train_steps
        ):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, global_step)
            latest_saved_step = global_step

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if latest_saved_step:
        final_path = update_final_checkpoint_link(config.checkpoint_dir, config.checkpoint_dir / str(latest_saved_step))
        logging.info("Updated final checkpoint link: %s", final_path)
    wandb.finish()


def main(args: argparse.Namespace) -> None:
    init_logging()
    set_seed(args.seed)
    config, data_factory = build_train_config(args)
    logging.info("Using OpenPI root: %s", os.environ.get("OPENPI_ROOT", str(DEFAULT_OPENPI_ROOT)))
    logging.info("Using LeRobot root: %s", os.environ.get("LEROBOT_ROOT", str(DEFAULT_LEROBOT_ROOT)))
    logging.info("Checkpoint dir: %s", config.checkpoint_dir)
    logging.info("Global batch size: %s", config.batch_size)
    logging.info("FSDP devices: %s", config.fsdp_devices)
    logging.info("Model image keys: %s", config.model.image_keys)
    logging.info("Model action dim: %s, dataset action dim: %s", config.model.action_dim, ACTION_DIM)
    logging.info("Freeze filter: %s", config.freeze_filter)

    if args.dry_run_batches > 0:
        dry_run(config, data_factory, args.dry_run_batches, args.add_action_noise)
        return

    train(config, data_factory, add_action_noise=args.add_action_noise)


if __name__ == "__main__":
    main(parse_args())

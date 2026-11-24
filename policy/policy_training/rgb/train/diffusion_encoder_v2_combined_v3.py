import argparse
import gc
from pathlib import Path

import torch
import wandb
from lerobot.configs.types import NormalizationMode
from lerobot.policies.diffusion_encoder_finetune_v2.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion_encoder_finetune_v2.modeling_diffusion import DiffusionPolicy
from tqdm import tqdm

from policy.policy_training.rgb.train.common import (
    build_combined_v4_rgb_dataloader,
    init_wandb_run,
    load_policy_and_task_config,
    load_rgb_dataset_stats_and_features,
    log_train_loss,
    make_ema_model,
    parse_combined_rgb_train_args,
    prepare_rgb_batch,
    reset_checkpoint_dir,
    rgb_camera_keys,
    rgb_delta_timestamps,
    rgb_policy_features,
    rgb_required_columns,
    set_seed,
)
from policy.policy_training.rgb.diffusion import (
    make_or_load_diffusion_processors,
    save_diffusion_checkpoint,
)
from policy.policy_training.utils import abs_to_rel_actions, inject_noise, update_relative_action_stats
from keypoint_program_solver.ik import KinHelper
from keypoint_program_solver.utils import resolve_robot_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parse_combined_rgb_train_args(parser)
    return parser.parse_args()


def main(args):
    config, task_cfg = load_policy_and_task_config(
        args.policy_config_path,
        args.tasks_config_path,
        "diffusion_encoder_v2",
        args.task,
    )
    init_wandb_run(
        project="diffusion_encoder_v2_training",
        config=config,
        run_name=(
            f"diffusion_encoder_v2_rgb_combined_"
            f"{Path(args.ckpt_save_dir).parent.name}_{Path(args.ckpt_save_dir).name}"
        ),
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
    camera_keys = rgb_camera_keys(task_cfg)

    set_seed(int(args.seed))

    dataset_stats, dataset_features = load_rgb_dataset_stats_and_features(
        repo_id,
        args.dataset_dir,
    )
    input_feature_cols, input_features, output_features = rgb_policy_features(
        dataset_features,
        camera_keys,
    )

    policy_config = DiffusionConfig(
        n_obs_steps=n_obs_steps,
        horizon=chunk_size,
        n_action_steps=n_action_steps,
        input_features=input_features,
        output_features=output_features,
        crop_shape=None,
        device=args.device,
        normalization_mapping={
            "VISUAL": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.ABSOLUTE_STATE_MIN_MAX_V1,
            "ACTION": (NormalizationMode.RELATIVE_ACTION if use_relative_action
                       else NormalizationMode.ABSOLUTE_ACTION_MIN_MAX_V1),
        },
    )

    delta_timestamps = rgb_delta_timestamps(
        policy_infer_freq,
        n_obs_steps,
        chunk_size,
        action_timestamp_offset,
        camera_keys,
    )
    required_columns = rgb_required_columns(camera_keys)
    dataset, dataloader = build_combined_v4_rgb_dataloader(
        required_columns=required_columns,
        recovery_dataset_path=args.recovery_dataset_path,
        new_recovery_episode_count=args.new_recovery_episode_count,
        repo_id=repo_id,
        dataset_dir=args.dataset_dir,
        delta_timestamps=delta_timestamps,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_ratio_cfg=dataset_ratio_cfg,
        seed=int(args.seed),
    )

    if use_relative_action:
        dataloader_normal = torch.utils.data.DataLoader(
            dataset.base_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        update_relative_action_stats(
            stats=dataset_stats,
            horizon=chunk_size,
            dataloader=dataloader_normal,
        )
        del dataloader_normal
        gc.collect()

    preprocessor, postprocessor = make_or_load_diffusion_processors(
        policy_config,
        dataset_stats,
        args.pretrained_ckpt_path,
    )

    if args.pretrained_ckpt_path is not None:
        policy = DiffusionPolicy.from_pretrained(
            pretrained_name_or_path=args.pretrained_ckpt_path,
            config=policy_config,
        )
        num_steps = int(config["train"]["num_steps_finetune"])
        save_freq = int(config["train"]["save_freq_finetune"])
    else:
        policy = DiffusionPolicy(config=policy_config)
        num_steps = int(config["train"]["num_steps_rgb"])
        save_freq = int(config["train"]["save_freq"])
    policy.to(args.device)
    policy.train()

    ema = make_ema_model(policy, config["ema"])

    optimizer_cfg = policy_config.get_optimizer_preset()
    optimizer = optimizer_cfg.build(policy.get_optim_params())

    ckpt_dir = reset_checkpoint_dir(args.ckpt_save_dir)

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

                batch = prepare_rgb_batch(
                    batch,
                    device=args.device,
                    input_feature_cols=input_feature_cols,
                    camera_keys=camera_keys,
                    no_ee=args.no_ee,
                    add_single_observation_step=n_obs_steps == 1,
                    squeeze_single_observation_step=False,
                )
                batch = preprocessor(batch)
                loss, loss_dict = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.step(policy)

                global_step += 1
                recycled_steps += 1
                log_train_loss(loss, loss_dict, global_step)
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % save_freq == 0:
                    checkpoint_policy = ema.averaged_model if ema is not None else policy
                    save_diffusion_checkpoint(
                        checkpoint_policy,
                        preprocessor,
                        postprocessor,
                        ckpt_dir / f"step_{global_step}",
                    )
        finally:
            del dataloader_iter
            gc.collect()

    print("Training finished. Saving final checkpoint...")
    checkpoint_policy = ema.averaged_model if ema is not None else policy
    save_diffusion_checkpoint(
        checkpoint_policy,
        preprocessor,
        postprocessor,
        ckpt_dir / "final",
    )
    progress_bar.close()
    wandb.finish()


if __name__ == "__main__":
    main(parse_args())

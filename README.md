# UniKey: Keypoints as a Unified Interface for Scalable Data Generation and Policy Learning

**Authors**

Yuan Fang¹, Yixuan Wang¹, Peng-Hao Hsu¹, Richard Cheng², Changxi Zheng¹, Katherine Liu², and Yunzhu Li¹

¹ Columbia University · ² Toyota Research Institute

[Project website](https://yfang777.github.io/UniKey_Blog/)

UniKey generates robot manipulation demonstrations in MuJoCo and uses them to train RGB policies. An xArm7 follows a task program, a geometric judge checks the trajectory, and two cameras record the successful episode. The policy training code comes from Codegen_Datagen.

| Task | What the robot does | Keypoints |
| --- | --- | ---: |
| `hang_mug` | Hang a mug on a rack | 13 |
| `arrange_table` | Arrange a fork and spoon around a bowl | 15 |
| `flip_then_arrange_shoe` | Flip the right shoe and place it beside the left shoe on a tray | 18 |

## Install

Use Linux x86-64 with Python 3.12, FFmpeg shared libraries, and an EGL-capable graphics driver. From a fresh clone, install the CPU environment and enable headless rendering.

```bash
git clone https://github.com/yfang777/UniKey.git
cd UniKey
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
```

Run the following commands from the repository root.

## Generate

Start with one mug demonstration and validate the saved trajectory and both videos.

```bash
python scripts/demo_hang_mug.py
python -m policy.utils.validate_dataset data/demo_hang_mug/dataset
```

To generate a dataset for any task in the table, choose a new output directory. This example collects ten successful episodes using two workers.

```bash
python -m policy.data_generation.data_generate \
  --task arrange_table --episodes 10 --workers 2 --seed 0 \
  --output data/arrange_table
```

The default `--mode stable` replays the task’s recorded targets. Use `--mode normal` to solve the motion stages from keypoint constraints. Each output contains the sampled scenes in `scenes/` and a 50 Hz LeRobot dataset in `dataset/`, including actions, observations, camera videos, and MuJoCo states for replay. Generated scene XML uses local asset paths, so regenerate scenes after moving the checkout.

## Train an ACT policy

Dense sampling collects complete expert episodes and trains a policy on them. The example below names the checkpoint by its 3,000-frame target. The actual frame count can differ because the generator keeps episodes whole.

```bash
python -m scripts.policy_training_dense_sampling \
  --task hang_mug --policy ACT --device cpu --seed 1 --save_dir outputs \
  --start_dataset_step_number 3000 --total_dataset_step_number 3000 \
  --incremental_dataset_step_number 1000 --start_training_steps 1000 \
  --datagen_workers 2 --training_workers 0 --only_generate_training
```

Corrective training starts from that checkpoint. It rolls out the policy, detects where it diverges from the expert, restores the simulator there, and collects an expert recovery. The fine-tuner mixes recovery frames with the original demonstrations.

```bash
python -m scripts.policy_training_corrective \
  --task hang_mug --policy ACT --device cpu --seed 1 --save_dir outputs \
  --start_dataset_step_number 3000 --total_dataset_step_number 4000 \
  --incremental_dataset_step_number 1000 --incremental_training_steps 100 \
  --training_workers 0 --rollout_batch_size 1 --test_scene_count 1
```

Keep the task, policy, seed, and save directory the same across both commands. Dense datasets and checkpoints are under `outputs/hang_mug/dense_sample/`. Recovery clips, fine-tuned checkpoints, and evaluation results are under `outputs/hang_mug/corrective/`. Both scripts accept `--help` for resume and evaluation options.

ACT uses the standard installation above. Diffusion training uses the pinned [Codegen LeRobot fork](https://github.com/yfang777/Codegen_lerobot_v0.5.1/tree/ee666d9406af38f5f0ebf8d472eaf1ef313e509f). Pi05 also requires JAX, OpenPI training dependencies, and pretrained weights. See the [Pi05 configuration](configurations/policy/Pi05.yaml) and the [Codegen OpenPI revision](https://github.com/yfang777/Codegen_Openpi/tree/f0932d9c6f5dccd28851078c72da15f942080c1a).

## Repository guide

- [Task configuration](configurations/tasks.yaml) defines the robot, cameras, and judge for each task. [Task programs](assets/solution/) contain the constraints and recorded targets.
- [Keypoint program solver](keypoint_program_solver/README.md) documents both solver modes.
- [Policy code](policy/) contains scene sampling, dataset generation, training, evaluation, and corrective collection.
- [Validation results](VALIDATION.md) summarize the tested workflows. [Third-party notices](THIRD_PARTY_NOTICES.md) list code and asset sources.

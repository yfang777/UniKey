# UniKey

A minimal simulation data-generation example for robot manipulation with relational
keypoints. An xArm7 picks up a colored cube and places it on the matching plate.
The pipeline samples a MuJoCo scene, executes recorded targets relative to the
objects, checks placement constraints, and exports a local LeRobot v3 dataset.

This release contains one task, `arrange_cube_by_color`, including its robot,
meshes, camera calibration, configurations, and exporter. No hardware, API keys,
model weights, private submodules, or dataset downloads are needed. Generation
never uploads anything.

## Installation

Use Linux x86-64 and Python 3.12. Headless rendering needs an EGL-capable driver;
the tested machine has an NVIDIA GPU. Ubuntu packages `libegl1`, `libgl1`, and
`ffmpeg` provide rendering/video libraries. CPU PyTorch is sufficient.

Download/extract or clone this repository, then run from its root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

A Conda environment with Python 3.12 also works with these pip commands.
The package named `pin` supplies Pinocchio.

## Generate one episode

```bash
export MUJOCO_GL=egl
python -m scene_generator.scene_generate \
  --base_dir data/example/scenes --num 1 --seed 0
python -m scripts.datagen_single \
  --scene_list data/example/scenes/sample_0/tmp_scene.xml \
  --root_path data/example/dataset
python -m scripts.validate_dataset data/example/dataset
```

Output directories must be new; existing data is never overwritten. Choose new
directories to repeat a run. Use `MUJOCO_GL=glfw` for a local graphical session.
Regenerate scenes after moving the checkout because XML files reference its assets.

For a batch, increase `--num`, then pass
`--scene_list data/example/scenes/sample_*/tmp_scene.xml`. Each scene is attempted
once. Successful episodes are saved; `generation.json` records every judged scene.
A run with no successful episodes exits with an error.

## Output

The published LeRobot `0.5.1` package writes a v3 dataset:

```text
data/example/dataset/
  data/chunk-000/file-000.parquet
  videos/observation.images.front/...
  videos/observation.images.wrist/...
  meta/info.json
  meta/episodes/...
  meta/state_buffers/chunk-000/file-000.parquet
  generation.json
```

Frames are recorded at 50 Hz. State and action each contain 10 values: XYZ,
6D rotation, and gripper state/action. Eight XYZ keypoints and a 10-value
constraint context record the matching plate and initial TCP/finger positions.
Front and wrist RGB videos are 640×480 and 1280×720. Simulation state buffers
are stored separately for replay.

The judge checks whether the cube reaches the selected plate during the
trajectory. It does not independently judge grasp, release, or final resting
stability. Release and return motions still execute.

## Structure

- `scene_generator/`: seeded sampling and MuJoCo scene assembly.
- `rekep_solver/`: execution, kinematics, constraints, and simulation.
- `robot/`: model paths and cameras.
- `scripts/`: dataset export and validation.
- `configurations/tasks.yaml`: physics, camera calibration, and execution settings.
- `configurations/initializations_range/`: object sampling ranges.
- `assets/solution/arrange_cube_by_color/`: constraints, judge, and recorded targets.

The quickstart uses `StableExecutor` with recorded object-relative targets.
The retained `Executor` constraint optimizer additionally requires
`python -m pip install -r requirements-solver.txt`; this optional path is outside
the quickstart validation.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests load the scene/assets, check camera/robot metadata, verify output protection,
and exercise the selected-plate judge. The dataset validator checks trajectories,
metadata, state-buffer alignment, and both videos. See [VALIDATION.md](VALIDATION.md).

## Attribution and license

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A project-wide license for
original code has not yet been specified. Third-party notices retain their terms.

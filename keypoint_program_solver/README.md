# Keypoint program solver

Create a `Solver(task_cfg, config)` from `keypoint_program_solver.solver`. Both methods return a MuJoCo environment with the episode frames in `env.lerobot_features`.

- `solver.execute(scene, seed=0)` reads the task’s `constraints.py` and optimizes the end-effector pose for each movement stage using the current keypoints.
- `solver.execute_stable(scene)` replays the recorded stage targets in the task’s `sol.py`.

For dataset generation, pass `--mode normal` to optimize constraints or use the default `--mode stable` for replay. See the [generation commands](../README.md#generate) and [motion configuration](../configurations/keypoint_program_solver/config.yaml).

Shared camera, robot-model, and episode-context helpers live in `utils.py`. Task programs use the geometric costs in `constraint_helpers.py`.

import importlib.util
import os
import sys
import numpy as np
from rekep_solver.episode_constraints import bind_episode_constraints

def load_module(path: str, name: str):
    """Dynamically load a python module from a file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module not found at: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module '{name}' from '{path}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def bind_helpers(constraints_mod, helpers_mod):
    """Inject helper functions into the constraint module."""
    for k, v in helpers_mod.__dict__.items():
        if callable(v) and k.startswith("_"):
            setattr(constraints_mod, k, v)

def get_keypoints_array(row):
    if isinstance(row, (list, tuple)) or hasattr(row, '__array__'):
        arr = np.array(row, dtype=float)
        return arr.reshape(-1, 3)
    return np.zeros((0, 3))

def check_geometric_constraints(constraints_mod, keypoints, stage, tolerance):
    """`stage` is 1-indexed and matches the `stageN_constraintM` naming in
    the compiled constraint module. Returns `(passed, has_constraints)`:
    `has_constraints` is False when no `stage{stage}_constraint*` symbols
    exist (e.g. grasp/release stages); in that case `passed` is True by
    convention but the caller should not credit the stage as completed.
    """
    func_prefix = f"stage{stage}_constraint"
    has_constraints = False
    for name, func in constraints_mod.__dict__.items():
        if callable(func) and name.startswith(func_prefix):
            has_constraints = True
            cost = float(func(keypoints))
            if cost > tolerance:
                return False, True
    return True, has_constraints

def evaluate_episode(parquet, constraints, grasp_stage_actions, tolerance):
    """All stage indices are 1-indexed, matching `stageN_constraintM` in the
    compiled constraint module and the executor's stage loop.

    A constraint stage advances/credits when its `stage{stage}_constraint*`
    all pass at the current frame.

    No-constraint stages (grasp/release) are skipped silently: just
    `current_stage += 1`, no credit.

    Returns `(passed, last_success_stage, last_success_frame)` where
    `last_success_stage` is the 1-indexed most recently completed
    geometric stage (0 when nothing has passed yet).
    """
    episode_module = constraints
    kp_col = "observation.keypoints"
    num_stages = constraints.num_stages
    last_success_frame = 0
    last_success_stage = 0
    current_stage = 1

    if kp_col not in parquet.columns:
        return False, None, None

    for frame_idx, row in enumerate(parquet[kp_col]):
        if getattr(episode_module, "uses_episode_context", False):
            constraints = bind_episode_constraints(episode_module, parquet["observation.constraint_context"].iloc[frame_idx])
        keypoints = get_keypoints_array(row)

        while current_stage <= num_stages:
            geo_pass, has_constraints = check_geometric_constraints(
                constraints, keypoints, current_stage, tolerance,
            )
            if not has_constraints:
                current_stage += 1
                if current_stage > num_stages:
                    return True, last_success_stage, last_success_frame
                continue
            if geo_pass:
                last_success_stage = current_stage
                last_success_frame = frame_idx
                current_stage += 1
                if current_stage > num_stages:
                    return True, last_success_stage, last_success_frame
            else:
                break

    return False, last_success_stage, last_success_frame

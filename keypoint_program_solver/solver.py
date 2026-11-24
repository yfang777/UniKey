"""Execute motion constraints and recorded pose programs in MuJoCo."""

import runpy
from types import SimpleNamespace

import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.transform import Rotation

from . import constraint_helpers
from .env import ReKepEnv
from . import transform_utils as T
from .utils import bind_episode_constraints, spline_interpolate_poses


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    return T.pose2mat((pose[:3], pose[3:]))


def matrix_pose(matrix):
    return np.concatenate(T.mat2pose(matrix)).tolist()


def body_world_pose(env, name):
    body = env.data.body(name)
    return np.concatenate([body.xpos, body.xquat[[1, 2, 3, 0]]]).tolist()


def resolve_ee_target(stage, ee_pose, object_poses, initial_object_poses):
    """Compute one world-frame EE target from the stage-start observation."""
    if stage['type'] == 'lift':
        target = np.array(ee_pose, dtype=float, copy=True)
        target[2] = stage['target_z']
        return target
    if stage['type'] == 'lift_relative':
        target = np.array(ee_pose, dtype=float, copy=True)
        target[2] += stage['delta_z']
        return target
    target = pose_matrix(stage['target_pose'])
    if stage['type'] == 'fixed':
        world_target = target
    elif stage['type'] == 'initial_ee_relative':
        world_target = pose_matrix(initial_object_poses[stage['object']]) @ target
    elif stage['type'] == 'ee_relative':
        world_target = pose_matrix(object_poses[stage['object']]) @ target
    elif stage['type'] == 'object_relative':
        object_ee = np.linalg.inv(pose_matrix(object_poses[stage['object']])) @ pose_matrix(ee_pose)
        world_target = pose_matrix(object_poses[stage['reference']]) @ target @ object_ee
    else:
        raise ValueError(f"Unknown motion type: {stage['type']}")
    return np.asarray(matrix_pose(world_target))


def _solve_target(current, keypoints, movable, constraints, bounds, rng):
    """Minimize stage violations while keeping the target near the current pose."""
    # Rotate movable points about the TCP in world coordinates.
    offsets = keypoints[movable] - current[:3]
    predicted = keypoints.copy()

    def violations(x):
        predicted[movable] = Rotation.from_rotvec(x[3:]).apply(offsets) + x[:3]
        return np.array([constraint(predicted) for constraint in constraints])

    def objective(x):
        motion = np.linalg.norm(x[:3] - current[:3]) + 5 * np.linalg.norm(x[3:])
        return motion + 200 * np.maximum(violations(x), 0).sum()

    start = np.r_[current[:3], np.zeros(3)]
    start = np.clip(start, *np.asarray(bounds).T)
    local = {"method": "SLSQP", "bounds": bounds, "options": {"maxiter": 200, "ftol": 1e-9}}
    result = minimize(objective, start, **local)
    if not np.all(violations(result.x) <= 0.0005):
        result = dual_annealing(objective, bounds, x0=result.x, minimizer_kwargs=local,
                                maxfun=2000, rng=rng)
    rotation = Rotation.from_rotvec(result.x[3:]) * Rotation.from_quat(current[3:])
    return np.r_[result.x[:3], rotation.as_quat()]


class Solver:
    """Run either execution mode in a fresh environment for each scene."""

    def __init__(self, task_cfg, config):
        self.task_cfg = task_cfg
        self.config = config

    def make_environment(self, scene):
        return ReKepEnv(scene, self.task_cfg["max_ee_linear_vel"], self.config,
                        self.task_cfg["dummy_simulate_t"])

    def _move(self, env, current, target):
        path = spline_interpolate_poses(np.stack([current, target]),
                                        self.config["interpolate_pos_step_size"],
                                        self.config["interpolate_rot_step_size"])
        for index, pose in enumerate(path):
            env.execute_action(np.r_[pose, env.gripper_null_action_value], precise=index == len(path) - 1)

    @staticmethod
    def _gripper(env, action):
        if action == "open":
            env.open_gripper()
        elif action == "close":
            env.close_gripper()
        else:
            raise ValueError(f"Unknown gripper action {action!r}")

    @staticmethod
    def _capture_gripper(env):
        gripper = env.get_keypoint_positions()[[0, -2, -1]].reshape(-1)
        env.episode_context = np.concatenate([env.episode_context[:24], gripper])

    def execute_stable(self, scene):
        """Replay the task's recorded stage targets and return the recorded environment."""
        env = self.make_environment(scene)
        stages = runpy.run_path(self.task_cfg["stable_solution_path"])["STAGES"]
        self.run_stages(env, stages)
        env.lerobot_features.append(env.get_lerobot_feature())
        return env

    def run_stages(self, env, stages, *, initial=None):
        """Run recorded stages from the current simulator state."""
        if initial is None:
            initial = {name: body_world_pose(env, name) for name in env.object_map}
        for stage in stages:
            if stage.get("capture_gripper_context", False):
                self._capture_gripper(env)
            if stage["type"] == "gripper":
                self._gripper(env, stage["action"])
            else:
                current = env.get_ee_pose()
                objects = {name: body_world_pose(env, name) for name in env.object_map}
                self._move(env, current, resolve_ee_target(stage, current, objects, initial))
            if "settle_s" in stage:
                env.step(env.data.ctrl[:9].copy(), n=int(np.ceil(stage["settle_s"] / env.dt)))

    def execute(self, scene, *, seed=0):
        """Optimize the task's motion constraints against observed keypoints at each stage."""
        env = self.make_environment(scene)
        helpers = {name: fn for name, fn in vars(constraint_helpers).items()
                   if name.startswith("_") and callable(fn)}
        program = SimpleNamespace(**runpy.run_path(self.task_cfg["constraint_path"], init_globals=helpers))
        rng = np.random.default_rng(seed)
        bounds = list(zip(env.bounds_min, env.bounds_max)) + [(-np.pi, np.pi)] * 3
        candidate_body = None
        for stage, action in enumerate(program.grasp_stage, start=1):
            if stage in getattr(program, "capture_gripper_context_stages", ()):
                self._capture_gripper(env)
            if action == 1:
                self._gripper(env, "close")
                candidate_body = env.keypoints[program.grasp_keypoints[stage - 1]]["obj_name"]
            elif action == -1:
                self._gripper(env, "open")
                candidate_body = None
            elif action == 0:
                bound = bind_episode_constraints(program, env.episode_context)
                prefix = f"stage{stage}_constraint"
                constraints = [fn for name, fn in vars(bound).items()
                               if name.startswith(prefix) and name[len(prefix):].isdigit() and callable(fn)]
                if not constraints:
                    raise ValueError(f"No motion constraints defined for stage {stage}")
                keypoints = env.get_keypoint_positions()
                movable = np.zeros(len(keypoints), dtype=bool)
                movable[[0, -2, -1]] = True
                if candidate_body is not None and env.finger_contacts_body(candidate_body):
                    for index, point in env.keypoints.items():
                        if point["obj_name"] == candidate_body:
                            movable[index] = True
                current = env.get_ee_pose()
                target = _solve_target(current, keypoints, movable, constraints, bounds, rng)
                self._move(env, current, target)
            else:
                raise ValueError(f"Unknown grasp_stage action {action} at stage {stage}")
        env.lerobot_features.append(env.get_lerobot_feature())
        return env

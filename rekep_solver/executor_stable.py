"""Execute recorded rigid transforms without subgoal optimization.

Poses are [x, y, z, qx, qy, qz, qw]. T_A_B maps B coordinates into A.
Object frames are raw MuJoCo body frames, not ReKepEnv.get_object_pose().
"""

import runpy
import time

import numpy as np

from . import transform_utils as T
from .env import ReKepEnv, StepBudgetExceeded
from .executor import Executor


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    return T.pose2mat((pose[:3], pose[3:]))


def matrix_pose(matrix):
    return np.concatenate(T.mat2pose(matrix)).tolist()


def body_world_pose(env, name):
    body = env.data.body(name)
    return np.concatenate([body.xpos, body.xquat[[1, 2, 3, 0]]]).tolist()


def site_world_poses(env):
    """Record named keypoint-site frames separately from object body frames."""
    poses = {}
    for names in env.site_map.values():
        for name in names:
            site = env.data.site(name)
            transform = np.eye(4)
            transform[:3, :3] = site.xmat.reshape(3, 3)
            transform[:3, 3] = site.xpos
            poses[name] = matrix_pose(transform)
    return poses


def resolve_ee_target(stage, ee_pose, object_poses, site_poses=None, initial_object_poses=None):
    """Compute one world-frame EE target from the stage-start observation."""
    if stage['type'] == 'lift':
        target = np.array(ee_pose, dtype=float, copy=True)
        target[2] = stage['target_z']
        return target
    if stage['type'] == 'lift_relative':
        target = np.array(ee_pose, dtype=float, copy=True)
        target[2] += stage['delta_z']
        return target
    if stage['type'] == 'site_lift':
        target = np.array(ee_pose, dtype=float, copy=True)
        target[2] += stage['target_z'] - site_poses[stage['site']][2]
        return target
    target = pose_matrix(stage['target_pose'])
    if stage['type'] == 'fixed':
        world_target = target
    elif stage['type'] == 'initial_ee_relative':
        world_target = pose_matrix(initial_object_poses[stage['object']]) @ target
    elif stage['type'] == 'ee_relative':
        world_target = pose_matrix(object_poses[stage['object']]) @ target
    elif stage['type'] in ('object_relative', 'site_relative'):
        object_ee = np.linalg.inv(pose_matrix(object_poses[stage['object']])) @ pose_matrix(ee_pose)
        references = site_poses if stage['type'] == 'site_relative' else object_poses
        world_target = pose_matrix(references[stage['reference']]) @ target @ object_ee
    else:
        raise ValueError(f"Unknown motion type: {stage['type']}")
    return np.asarray(matrix_pose(world_target))


def extract_stages(stage_spec, snapshots):
    """Extract achieved stage-end targets from one optimizer demonstration."""
    stages = []
    for spec in stage_spec:
        if spec['type'] in ('gripper', 'lift', 'lift_relative', 'site_lift'):
            stages.append(dict(spec))
            continue
        snapshot = snapshots[spec['stage']]
        objects = snapshot['object_poses']
        if spec['type'] == 'fixed':
            target = pose_matrix(snapshot['ee_pose'])
        elif spec['type'] == 'initial_ee_relative':
            initial = snapshots[1]['start_object_poses'][spec['object']]
            target = np.linalg.inv(pose_matrix(initial)) @ pose_matrix(snapshot['ee_pose'])
        elif spec['type'] == 'ee_relative':
            target = np.linalg.inv(pose_matrix(objects[spec['object']])) @ pose_matrix(snapshot['ee_pose'])
        elif spec['type'] in ('object_relative', 'site_relative'):
            references = snapshot['site_poses'] if spec['type'] == 'site_relative' else objects
            target = np.linalg.inv(pose_matrix(references[spec['reference']])) @ pose_matrix(objects[spec['object']])
        else:
            raise ValueError(f"Unknown motion type: {spec['type']}")
        stages.append({**spec, 'target_pose': matrix_pose(target)})
    return stages


class StableExecutor(Executor):
    """Reuse motion execution and gripper actions; no optimization components."""

    def __init__(self, max_ee_linear_vel, config_path='configurations/rekep_solver/config.yaml', record_images=True):
        super().__init__(max_ee_linear_vel, config_path)
        self.record_images = record_images
        self.stages = []

    def _init_components(self, scene_xml_path, save_dir, dummy_simulate_t=0.0):
        self.env = ReKepEnv(max_ee_linear_vel=self.max_ee_linear_vel)
        self.env.reset(scene_xml_path, dummy_simulate_t=dummy_simulate_t)
        if not self.record_images:
            self.env.cams = []

    def perform_task_fixed(self, scene_xml_path, module_filename, dummy_simulate_t=0.0):
        stages = runpy.run_path(str(module_filename))['STAGES']
        self._init_components(scene_xml_path, None, dummy_simulate_t)
        self.initial_object_poses = {name: body_world_pose(self.env, name) for name in self.env.object_map}
        return self._execute_stages(stages)


    def _execute_stages(self, stages):
        self.stages = []
        self.early_stopped = False
        targets = {}
        for stage in stages:
            if stage.get('capture_gripper_context', False):
                gripper = self.env.get_keypoint_positions()[[0, -2, -1]].reshape(-1)
                self.env.episode_context = np.concatenate([self.env.episode_context[:24], gripper])
            record = {
                'stage': stage['stage'], 'type': stage['type'],
                'start_frame': len(self.env.lerobot_features),
                'start_sim_time': float(self.env.data.time),
            }
            try:
                if stage['type'] == 'gripper':
                    record['action'] = stage['action']
                    if stage['action'] == 'close':
                        self._do_grasp()
                    elif stage['action'] == 'open':
                        self._do_release()
                    else:
                        raise ValueError(f"Unknown gripper action: {stage['action']}")
                else:
                    objects = {name: body_world_pose(self.env, name) for name in self.env.object_map}
                    target = resolve_ee_target(stage, self.env.get_ee_pose(), objects, site_world_poses(self.env),
                                               self.initial_object_poses)
                    targets[stage['stage']] = target.tolist()
                    record['target_ee_pose'] = target.tolist()
                    self._exec_path_actions(self._process_path(self.env.get_ee_pose(), [target]))
                if 'settle_s' in stage:
                    self.env.step(self.env.data.ctrl[:9].copy(),
                                  n=int(np.ceil(stage['settle_s'] / self.env.dt)))
            except StepBudgetExceeded:
                self.early_stopped = True
            record['end_frame_exclusive'] = len(self.env.lerobot_features)
            record['end_sim_time'] = float(self.env.data.time)
            record['ee_pose'] = self.env.get_ee_pose().tolist()
            record['object_poses'] = {name: body_world_pose(self.env, name) for name in self.env.object_map}
            record['site_poses'] = site_world_poses(self.env)
            self.stages.append(record)
            if self.early_stopped:
                break
        self.env.add_last_frame_lerobot_features()
        self.lerobot_features = self.env.lerobot_features
        self.env.step_budget = None
        return not self.early_stopped, targets

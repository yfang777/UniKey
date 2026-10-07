# executor_v2.py

import os, json, yaml, numpy as np, gc
from copy import deepcopy
from pathlib import Path

from . import constraint_helpers as ch
from .env import ReKepEnv
# from .ik_solver_lula import IKSolver
from .subgoal_solver import SubgoalSolver
from . import transform_utils as T
from .constraint_compile import ConstraintCompiler
from .utils import get_linear_interpolation_steps, spline_interpolate_poses


class Executor:
    """
    Stage machine:
      For stage s:
        solve constraints, execute
    """
    def __init__(self, max_ee_linear_vel: float, config_path="configurations/rekep_solver/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.env = None
        self.max_ee_linear_vel = max_ee_linear_vel
        self.max_retries_per_stage = self.config['main']['max_retries_per_stage']
        self.act_seq_num = self.config['main']['act_seq_num']

    def _init_components(self, scene_xml_path, save_dir, dummy_simulate_t=0.0):
        self.env = ReKepEnv(max_ee_linear_vel=self.max_ee_linear_vel)
        self.env.reset(scene_xml_path, dummy_simulate_t=dummy_simulate_t)
        # self.ik = IKSolver()
        self.ik = self.env.kin_helper
        self.subgoal_solver = SubgoalSolver(self.config['subgoal_solver'], self.ik, np.zeros(7))
        # self.path_solver    = PathSolver(self.config['path_solver'],    self.ik, np.zeros(7))

    def _process_path(self, curr_ee_pose, path):
        cps = np.concatenate([curr_ee_pose.reshape(1,-1), path], axis=0)
        # nsteps = get_linear_interpolation_steps(
        #     cps[0], cps[-1],
        #     self.config['main']['interpolate_pos_step_size'],
        #     self.config['main']['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(cps, self.config['main']['interpolate_pos_step_size'], self.config['main']['interpolate_rot_step_size'])
        act_seq = np.zeros((dense_path.shape[0], 8))
        act_seq[:,:7] = dense_path
        act_seq[:,7]  = self.env.gripper_null_action_value
        return act_seq

    def _exec_path_actions(self, act_seq):
        for i, action in enumerate(act_seq):
            precise = (i == len(act_seq)-1)
            self.env.execute_action(action, precise=precise)
        return True




    def perform_task_fixed(
        self,
        scene_xml_path: str,
        module_filename: str,
        dummy_simulate_t: float = 0.0,
    ) -> bool:
        if not os.path.isabs(scene_xml_path):
            scene_xml_path = os.path.abspath(scene_xml_path)
        if not os.path.isabs(module_filename):
            module_filename = os.path.abspath(module_filename)
        module_dir = str(os.path.dirname(module_filename))
        module_name = str(Path(module_filename).name)

        self._init_components(
            scene_xml_path,
            save_dir=None,
            dummy_simulate_t=dummy_simulate_t,
        )
        cc = ConstraintCompiler(
            module_dir=module_dir,
            module_name=module_name,
        )
        cc.bind_dependencies(
            _compose_vector=ch._compose_vector,
            _compose_unit_vector=ch._compose_unit_vector,
            _compose_plane_normal=ch._compose_plane_normal,
            _xy_project=ch._xy_project,
            _gripper_left_to_right=ch._gripper_left_to_right,
            _unit=ch._unit,
            _pos_cost=ch._pos_cost,
            _parallel_cost=ch._parallel_cost,
            _samedir_cost=ch._samedir_cost,
            _samedir_xy_cost=ch._samedir_xy_cost,
            _orthogonal_cost=ch._orthogonal_cost,
            _theta_cost=ch._theta_cost,
            _point_in_plane_xy_cost=ch._point_in_plane_xy_cost,
            _point_in_plane_yz_cost=ch._point_in_plane_yz_cost,
            _point_in_plane_xz_cost=ch._point_in_plane_xz_cost,
            _point_in_aabb_cost=ch._point_in_aabb_cost,
            _angle_30deg_from_neg_z_cost=ch._angle_30deg_from_neg_z_cost,
            _angle_60deg_from_neg_z_cost=ch._angle_60deg_from_neg_z_cost,
            _angle_75deg_from_neg_z_cost=ch._angle_75deg_from_neg_z_cost,
            np=np
        )
        loaded = cc.load_py(self.env.episode_context)
        constraints_all = loaded["constraints"]
        grasp_stage = loaded["grasp_stage"]
        num_stages = len(grasp_stage)
        grasp_keypoints = loaded["grasp_keypoints"]

        stage = 1
        solution = {}
        while stage <= num_stages:
            loaded = self._prepare_stage(cc, loaded, stage)
            constraints_all = loaded["constraints"]

            grasp_state = grasp_stage[stage - 1]
            is_grasp = (grasp_state == 1)
            is_release = (grasp_state == -1)

            if is_grasp:
                self._do_grasp()
                stage += 1
                continue
            elif is_release:
                self._do_release()
                stage += 1
                continue
            else:
                constraints = constraints_all[stage]


            # Solve subgoal
            action_done = False
            while not action_done:
                kp = self.env.get_keypoint_positions()
                ee = self.env.get_ee_pose()
                sdf = self.env.get_sdf_voxels(self.config['main']['sdf_voxel_size'])
                col = self.env.get_collision_points()
                joint = self.env.get_arm_joint_postions()

                # if stage < num_stages and grasp_keypoints[stage] != -1:
                #     object_name = self.env.get_object_by_keypoint(grasp_keypoints[stage])
                #     object_pose = self.env.get_object_pose(object_name)
                # else:
                #     object_pose = None

                subgoal_pose, _dbg1 = self.subgoal_solver.solve(
                    ee, kp, self._movable_mask(), constraints, [], sdf, col,
                    joint, from_scratch=True, object_pose=None
                )
                act_seq = self._process_path(ee, [subgoal_pose])

                solution[stage] = np.asarray(subgoal_pose).tolist()

                # Execute
                if len(act_seq) < self.act_seq_num:
                    self._exec_path_actions(act_seq)
                    action_done = True
                else:
                    self._exec_path_actions(act_seq[:self.act_seq_num])

            stage += 1


        self.env.add_last_frame_lerobot_features()
        self.lerobot_features = deepcopy(self.env.lerobot_features)
        # All stages done
        return True, solution






    # ---- helpers ----
    def _prepare_stage(self, compiler, loaded, stage):
        if stage in getattr(compiler.module, 'capture_gripper_context_stages', ()):
            gripper = self.env.get_keypoint_positions()[[0, -2, -1]].reshape(-1)
            self.env.episode_context = np.concatenate([self.env.episode_context[:24], gripper])
            return compiler.load_py(self.env.episode_context)
        return loaded

    def _movable_mask(self):
        # EE is always movable; other keypoints movable only if grasped
        kp = self.env.get_keypoint_positions()
        mask = np.zeros(kp.shape[0], dtype=bool)
        mask[0] = True
        for i in range(1, len(mask)):
            obj = self.env.get_object_by_keypoint(i)
            mask[i] = self.env.is_grasping(obj)
        mask[-2] = True
        mask[-1] = True
        return mask

    def _do_grasp(self):
        pre = self.env.get_ee_pose()
        pose = pre.copy()
        pose[:3] += T.quat2mat(pre[3:]) @ np.array([self.config['main']['grasp_depth'], 0, 0])
        action = np.concatenate([pose, [self.env.gripper_close_action_value]])
        self.env.execute_action(action, precise=True)

    def _do_release(self):
        self.env.open_gripper()


    def close(self):
        if self.env is not None:
            self.env.close()

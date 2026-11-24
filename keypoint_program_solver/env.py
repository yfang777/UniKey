"""MuJoCo motion execution and trajectory recording for the released tasks."""

import mujoco
import numpy as np

from .ik import KinHelper
from . import transform_utils as T
from .utils import (
    resolve_robot_paths, scene_model_text,
    angle_between_quats, angle_between_rotmat, get_linear_interpolation_steps,
    linear_interpolate_poses, get_body_site_map, get_object_map, bcolors, get_clock_time,
)


class ReKepEnv:
    def __init__(self, scene, velocity, config, settle_time):
        self.max_ee_linear_vel = velocity
        self.max_ee_angular_vel = config["max_ee_angular_vel"]
        self.interpolate_pos_step_size = config["interpolate_pos_step_size"]
        self.interpolate_rot_step_size = config["interpolate_rot_step_size"]
        self.bounds_min = np.asarray(config["bounds_min"])
        self.bounds_max = np.asarray(config["bounds_max"])
        self.gripper_open_action_value = 1
        self.gripper_close_action_value = 0
        self.gripper_null_action_value = -1
        self.control_hz = 50
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.robot_variant = scene_model_text(self.model, "robot_variant")
        self.cams = scene_model_text(self.model, "observation_cameras").split()
        urdf, _ = resolve_robot_paths(self.robot_variant)
        self.kin_helper = KinHelper(urdf)
        self.site_map = get_body_site_map(scene)
        self.object_map = get_object_map(scene)
        self.last_gripper_status = 1
        home = [0., -0.78539816, 0., 0.52359878, 0., 1.30899694, 0.]
        self.data.qpos[:7] = home
        self.data.ctrl[:7] = home
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=round(settle_time / self.dt))
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.lerobot_features = []
        self.render_qpos = self.data.qpos.copy()
        self.episode_context = None
        if "shoe_tray" in self.object_map:
            keypoints = self.get_keypoint_positions()
            gripper = keypoints[[0, -2, -1]].reshape(-1)
            self.episode_context = np.concatenate([keypoints[1:6].reshape(-1), gripper, gripper])

    def step(self, action, n=1):
        self.data.ctrl[:9] = action
        for _ in range(n):
            if self.data.time >= len(self.lerobot_features) / self.control_hz:
                self.lerobot_features.append(self.get_lerobot_feature())
            # mj_step computes the camera and geom transforms before integrating qpos.
            self.render_qpos[:] = self.data.qpos
            mujoco.mj_step(self.model, self.data)

    def get_ee_pose(self):
        """Get the pose of the end effector x, y, z, qx, qy, qz, qw"""
        link_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'link_tcp')
        link_ee_pos = self.data.site_xpos[link_ee_id]
        link_ee_quat = T.mat2quat(self.data.site_xmat[link_ee_id].reshape(3, 3))
        return np.concatenate([link_ee_pos, link_ee_quat])

    def get_ee_pose_mat(self):
        """Get the pose of the end effector in 4 * 4 matrix"""
        link_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'link_tcp')
        link_ee_pos = self.data.site_xpos[link_ee_id]
        link_ee_R = self.data.site_xmat[link_ee_id].reshape(3, 3)
        transformation = np.eye(4)
        transformation[:3, :3] = link_ee_R
        transformation[:3, 3] = link_ee_pos
        return transformation

    def get_lerobot_feature(self):

        features = {"render_qpos": self.render_qpos.copy()}

        ee_pose_mat = self.get_ee_pose_mat()
        pos = ee_pose_mat[:3, 3]
        rotation_6D = ee_pose_mat[:2, :3].reshape(6)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        qpos_index = self.model.jnt_qposadr[joint_id]
        gripper_norm = np.array([self.data.qpos[qpos_index]])
        features["observation.state"] = np.concatenate([pos, rotation_6D, gripper_norm])               # shape: (10,)

        action_qpos = self.data.qpos[:13].copy()
        action_qpos[0:7] = self.data.ctrl[0:7]
        action_ee_mat = self.kin_helper.compute_fk_from_link_names(
            action_qpos, ["link_tcp"]
        )["link_tcp"]
        action_pos = action_ee_mat[:3, 3]
        action_rotation_6D = action_ee_mat[:2, :3].reshape(6)
        action_gripper_norm = np.array([self.data.ctrl[7]])
        features["action"] = np.concatenate([action_pos, action_rotation_6D, action_gripper_norm])     # shape: (10,)

        for obj_name in self.object_map.keys():
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            obj_pos = self.data.xpos[obj_id]
            obj_quat = T.convert_quat(self.data.xquat[obj_id], to="xyzw")
            features[f"object.{obj_name}.pose"] = np.concatenate([obj_pos, obj_quat])

        features[f"observation.keypoints"] = self.get_keypoint_positions().reshape(-1)
        if self.episode_context is not None:
            features["observation.constraint_context"] = self.episode_context.copy()

        state_buf, last_gripper_status = self.save_state()
        features["state_buf"] = np.array(state_buf, dtype=np.float64)

        features["last_gripper_status"] = np.array([self.last_gripper_status], dtype=np.float64)
        return features

    def execute_action(
            self,
            action,
            precise=True,
        ):
            """
            Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

            Args:
                action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
                precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
            Returns:
                tuple: A tuple containing the position and rotation errors after reaching the target pose.
            """
            if precise:
                pos_threshold = 0.001
                rot_threshold = 2

                max_steps = 80  # convergence_step
            else:
                pos_threshold = 0.005
                rot_threshold = 1
                max_steps = 40

            intermediate_pos_threshold = 0.005
            intermediate_rot_threshold = 1

            action = np.array(action).copy()
            assert action.shape == (8,)
            target_pose = action[:7]
            gripper_action = action[7]

            if np.any(target_pose[:3] < self.bounds_min) \
                 or np.any(target_pose[:3] > self.bounds_max):

                target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

            current_pose = self.get_ee_pose()
            pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
            rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
            pos_is_close = pos_diff < self.interpolate_pos_step_size
            rot_is_close = rot_diff < self.interpolate_rot_step_size
            if pos_is_close and rot_is_close:

                pose_seq = np.array([target_pose])
            else:
                num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
                pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)

            total_steps = 0
            for pose in pose_seq[:-1]:
                steps = self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
                total_steps += steps

            pose = pose_seq[-1]
            steps = self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=max_steps)
            total_steps += steps

            pos_error, rot_error = self.compute_target_delta_ee(target_pose)

            if gripper_action == self.gripper_open_action_value:
                self.open_gripper()
            elif gripper_action == self.gripper_close_action_value:
                self.close_gripper()
            elif gripper_action == self.gripper_null_action_value:
                pass
            else:
                raise ValueError(f"Invalid gripper action: {gripper_action}")

            return pos_error, rot_error

    def open_gripper(self):
        """last_gripper_status: 1.0 = open, 0.0 = closed. Ctrl range: 0.0 = open, 0.85 = closed."""
        action = self.data.ctrl[:9].copy()
        start_gripper_values = action[7:9].copy()
        target_gripper_values = np.array([0.0, 0.0])

        gripper_open_time = 1.0
        gripper_steps = int(
            np.ceil(
                gripper_open_time
                * np.max(np.abs(target_gripper_values - start_gripper_values))
                / 0.85
                / self.dt
            )
        )
        for step_idx in range(1, gripper_steps + 1):
            alpha = step_idx / gripper_steps
            action[7:9] = (
                start_gripper_values
                + (target_gripper_values - start_gripper_values) * alpha
            )
            self.step(action, n=1)

        self.last_gripper_status = 1.0
        self.update_keypoints()

    def close_gripper(self):
        action = self.data.ctrl[:9].copy()
        start_gripper_values = action[7:9].copy()
        target_gripper_values = np.array([0.85, 0.85])

        gripper_close_time = 1.0
        gripper_steps = int(
            np.ceil(
                gripper_close_time
                * np.max(np.abs(target_gripper_values - start_gripper_values))
                / 0.85
                / self.dt
            )
        )
        for step_idx in range(1, gripper_steps + 1):
            alpha = step_idx / gripper_steps
            action[7:9] = (
                start_gripper_values
                + (target_gripper_values - start_gripper_values) * alpha
            )
            self.step(action, n=1)

        self.last_gripper_status = 0.0
        self.update_keypoints()

    def _move_to_waypoint(self, target_pose_world, pos_threshold=0.02, rot_threshold=3.0, max_steps=10):
        '''
        target_pose_world: x, y, z, qx, qy, qz, qw
        target_pose need to be convert into the gripper frame:
        '''

        pos_errors = []
        rot_errors = []
        count = 0

        steps_n = int(1 / self.dt * 1 / self.control_hz)
        step_duration = steps_n * self.dt

        while count < max_steps:
            reached, pos_error, rot_error = self._check_reached_ee(target_pose_world[:3], target_pose_world[3:7], pos_threshold, rot_threshold)
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)
            if reached:
                break
            target_mat_world = T.convert_pose_quat2mat(target_pose_world)

            current_pose = self.get_ee_pose()

            clamped_target_pose = self._clamp_ee_target(current_pose, target_pose_world, step_duration)

            target_mat_clamped = T.convert_pose_quat2mat(clamped_target_pose)

            qpos, success, error = self.kin_helper.compute_ik_from_mat(self.data.qpos[:13], target_mat_clamped)
            if self.last_gripper_status == 1.0:
                self.step(np.concatenate([qpos[0:7], [0, 0]]), n=steps_n)
            else:
                self.step(np.concatenate([qpos[0:7], [0.85, 0.85]]), n=steps_n)

            self.update_keypoints()

            count += 1
        if count == max_steps:
            print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose not reached after {max_steps} steps (pos_error: {pos_errors[-1].round(5)}, rot_error: {np.rad2deg(rot_errors[-1]).round(4)}){bcolors.ENDC}')

        return count

    def _clamp_ee_target(self, current_pose, target_pose, duration):
        """
        Calculates a new target pose that is within the velocity limits
        from the current pose given the time duration.
        """

        current_pos = current_pose[:3]
        target_pos = target_pose[:3]

        direction = target_pos - current_pos
        dist = np.linalg.norm(direction)

        max_dist = self.max_ee_linear_vel * duration

        if dist > max_dist:

            clamped_pos = current_pos + (direction / dist) * max_dist
        else:
            clamped_pos = target_pos

        current_quat = current_pose[3:]
        target_quat = target_pose[3:]

        angle_diff = angle_between_quats(current_quat, target_quat) # Returns radians
        max_angle = self.max_ee_angular_vel * duration

        if angle_diff > max_angle and angle_diff > 1e-6:
            t = max_angle / angle_diff

            q0 = current_quat
            q1 = target_quat

            dot = np.dot(q0, q1)
            if dot < 0.0:
                q1 = -q1
                dot = -dot

            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)

            if theta < 1e-6:
                clamped_quat = q1
            else:
                sin_theta = np.sin(theta)
                w1 = np.sin((1 - t) * theta) / sin_theta
                w2 = np.sin(t * theta) / sin_theta
                clamped_quat = (w1 * q0) + (w2 * q1)
                clamped_quat /= np.linalg.norm(clamped_quat) # Normalize
        else:
            clamped_quat = target_quat

        return np.concatenate([clamped_pos, clamped_quat])

    def _check_reached_ee(self, target_pos, target_xyzw, pos_threshold, rot_threshold):
        """
        this is supposed to be for true ee pose (franka hand) in robot frame
        """
        current_pose = self.get_ee_pose()
        current_pos = current_pose[:3]
        current_xyzw = current_pose[3:7]
        current_rotmat = T.quat2mat(current_xyzw)
        target_rotmat = T.quat2mat(target_xyzw)

        pos_diff = (target_pos - current_pos).flatten()
        pos_error = np.linalg.norm(pos_diff)

        rot_error = angle_between_rotmat(current_rotmat, target_rotmat)

        if pos_error < pos_threshold and rot_error < np.deg2rad(rot_threshold):

            return True, pos_error, rot_error

        return False, pos_error, rot_error

    def compute_target_delta_ee(self, target_pose):
        target_pos, target_xyzw = target_pose[:3], target_pose[3:]
        ee_pose = self.get_ee_pose()
        ee_pos, ee_xyzw = ee_pose[:3], ee_pose[3:]
        pos_diff = np.linalg.norm(ee_pos - target_pos)
        rot_diff = angle_between_quats(ee_xyzw, target_xyzw)

        return pos_diff, rot_diff

    def update_keypoints(self):
        """
        Register all <site> keypoints from the MuJoCo model, using provided body-to-site mapping.

        """
        self.keypoints = {}

        idx = 0
        for body_name, site_list in self.site_map.items():
            for site_name in site_list:
                if site_name in ['left_finger_site', 'right_finger_site']:
                    continue

                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                pos = self.data.site_xpos[site_id]

                self.keypoints[idx] = {
                    'obj_name': body_name,
                    'site_name': site_name,
                    'pos': pos
                }
                idx += 1

        finger_sites = [
            ('left_finger', 'left_finger_site'),
            ('right_finger', 'right_finger_site'),
        ]
        for body_name, site_name in finger_sites:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            pos = self.data.site_xpos[site_id]
            self.keypoints[idx] = {
                'obj_name': body_name,
                'site_name': site_name,
                'pos': pos
            }
            idx += 1

    def finger_contacts_body(self, name):
        """Check whether a finger currently touches the named task object."""
        body = self.model.body(name).id
        fingers = {self.model.body(side).id for side in ("left_finger", "right_finger")}
        for contact in self.data.contact:
            first, second = self.model.geom_bodyid[contact.geom]
            if (first == body and second in fingers) or (second == body and first in fingers):
                return True
        return False

    def get_keypoint_positions(self):
        """
        Returns keypoint positions from all registered keypoints.
        Shape: (N, 3)
        """
        self.update_keypoints()

        return np.array([k['pos'] for k in self.keypoints.values()])

    @staticmethod
    def _full_state_mask():
        s = mujoco.mjtState

        return (
            s.mjSTATE_TIME       |
            s.mjSTATE_QPOS       |
            s.mjSTATE_QVEL       |
            s.mjSTATE_ACT        |
            s.mjSTATE_WARMSTART  |
            s.mjSTATE_CTRL
        )

    def save_state(self) -> dict:
        """
        Save the entire MuJoCo sim state for exact restoration.
        Compatible with MuJoCo versions that don't define mjSTATE_ALL.
        """
        mask = self._full_state_mask()
        size = mujoco.mj_stateSize(self.model, mask)
        state_buf = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state_buf, mask)

        return state_buf, self.last_gripper_status

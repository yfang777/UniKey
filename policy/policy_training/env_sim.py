from keypoint_program_solver.utils import CameraRenderers
from keypoint_program_solver.utils import resolve_robot_paths, scene_model_text
from pathlib import Path
import cv2
import numpy as np
import mujoco
import mujoco.viewer

from keypoint_program_solver import transform_utils as T
from keypoint_program_solver.utils import get_object_map, get_body_site_map
from keypoint_program_solver.ik import KinHelper


# ------ Env ------
class SimEnv():
    def __init__(self, max_ee_linear_vel: float):
        self.dt = 1e-4
        self.policy_freq = 10
        self.control_freq = 50      # env step rate (1 / time advanced per env.step)
        self.lerobot_fps = 50       # lerobot_feature snapshot rate during physics
        self.t_horizon = 12

        self.render_mujoco = True
        self.render_width = 640
        self.render_height = 480
        self.viewer = None
        self.render_mujoco = True
        self.step_counter = 0
        self.use_relative_action = True

        # Velocity Control Config
        self.max_ee_linear_vel = max_ee_linear_vel
        self.max_ee_angular_vel = 2.0

        # Pointcloud Config
        self.pointcloud_bounds_min = np.array([0, -0.5, 0.01])
        self.pointcloud_bounds_max = np.array([0.8, 0.5, 0.9])
        self.pointcloud_sample_num = 1024

        self.min_depth = 0.1
        self.max_depth = 2.0

        self.reset_joint_pos = np.array([0., -0.78539816, 0., 0.52359878, 0., 1.30899694, 0.])

        self.renderer = None
        self.cam_ids = {}

    def reset(
        self,
        mujoco_xml_path,
        seed=None,
        options=None,
        log_video=False,
        camera_names=None,
        t_horizon=None,
        dummy_simulate_t=0.0,
    ):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        scene_path = Path(mujoco_xml_path)
        if scene_path.suffix == ".mjb":
            # Compiled randomized models retain texture pixels and camera perturbations.
            self.model = mujoco.MjModel.from_binary_path(str(scene_path))
            metadata_xml_path = scene_path.with_suffix(".xml")
        else:
            self.model = mujoco.MjModel.from_xml_path(str(scene_path))
            metadata_xml_path = scene_path
        self.robot_variant = scene_model_text(self.model, "robot_variant")
        urdf_path, _ = resolve_robot_paths(self.robot_variant)
        self.kin_helper = KinHelper(urdf_path)
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:7] = self.reset_joint_pos
        self.data.ctrl[0:7] = self.reset_joint_pos
        self.data.ctrl[7:9] = 0
        mujoco.mj_forward(self.model, self.data)

        if dummy_simulate_t < 0:
            raise ValueError("dummy_simulate_t must be >= 0")
        dummy_steps = int(round(dummy_simulate_t / self.dt))
        for _ in range(dummy_steps):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.site_map = get_body_site_map(str(metadata_xml_path))
        self.object_map = get_object_map(str(metadata_xml_path))

        self.step_counter = 0
        self.last_gripper_status = 1
        self.lerobot_features = []

        ee_pose_mat = self.get_ee_pose_mat()
        init_pos = ee_pose_mat[:3, 3]
        init_rot6d = ee_pose_mat[:2, :3].reshape(6)
        gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        gripper_qpos_idx = self.model.jnt_qposadr[gripper_joint_id]
        init_gripper = self.data.qpos[gripper_qpos_idx]
        self.last_action = np.concatenate(
            [init_pos, init_rot6d, [init_gripper]]
        ).astype(np.float64)

        self.log_video = log_video
        self.camera_names = camera_names or []

        if t_horizon is not None:
            self.t_horizon = t_horizon

        if self.log_video:
            self.renderer = CameraRenderers(self.model)
            self.cam_ids = {}
            for name in self.camera_names:
                self.cam_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)

        obs, _ = self.get_obs()
        self.update_keypoints()
        self.episode_context = None
        if "shoe_tray" in self.object_map:
            keypoints = self.get_keypoint_positions()
            gripper = keypoints[[0, -2, -1]].reshape(-1)
            self.episode_context = np.concatenate([keypoints[1:6].reshape(-1), gripper, gripper])

        return obs, {"initial_action": self.last_action.copy()}

    def step(self, action, no_render=False):
        assert action.shape == (13,)

        # Log the commanded action in 10-dim (pos + rot6d + gripper) form.
        gripper_clamped = min(float(action[12]), 0.85)
        self.last_action = np.concatenate(
            [action[0:3], action[3:9], [gripper_clamped]]
        ).astype(np.float64)

        # Gripper status: open: 1, close: 0
        if action[12] > 0.425:
            self.last_gripper_status = 0
        else:
            self.last_gripper_status = 1

        # Mujoco
        action_mat = np.eye(4)
        action_mat[:3, 3] = action[0:3]
        action_mat[:3, :3] = action[3:12].reshape(3, 3)

        poses_mat = [action_mat]

        for pose_mat in poses_mat:
            action_joint, success, error = self.kin_helper.compute_ik_from_mat(
                initial_qpos=self.data.qpos[:13], tf_mat=pose_mat
            )
            self.data.ctrl[0:7] = action_joint[0:7]

            gripper_value = action[12]
            self.data.ctrl[7:9] = np.array([gripper_value, gripper_value])

            for i in range(int(1 / self.dt / self.control_freq)):
                sim_time = self.data.time
                target_frame_time = len(self.lerobot_features) / self.lerobot_fps
                if sim_time >= target_frame_time:
                    self.lerobot_features.append(self.get_lerobot_features())

                mujoco.mj_step(self.model, self.data)
                self.step_counter += 1

        obs, _ = self.get_obs(no_render=no_render)

        reward, terminated, truncated = 0.0, False, False
        if self.step_counter >= self.t_horizon / self.dt:
            terminated = True

        info = {}
        return obs, reward, terminated, truncated, info

    def get_obs(self, no_render=False):
        obs = {}

        ee_pose_mat = self.get_ee_pose_mat()
        pos = ee_pose_mat[:3, 3]
        rotation_6D = ee_pose_mat[:2, :3].reshape(6)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        qpos_index = self.model.jnt_qposadr[joint_id]
        gripper_norm = np.array([self.data.qpos[qpos_index]])
        obs["observation.state"] = np.concatenate([pos, rotation_6D, gripper_norm])

        if self.log_video and not no_render:
            for camera_name in self.camera_names:
                cam_id = self.cam_ids[camera_name]
                renderer = self.renderer.for_camera(camera_name)
                renderer.disable_depth_rendering()
                renderer.disable_segmentation_rendering()
                renderer.update_scene(self.data, camera=cam_id)
                rgb_mj = renderer.render()
                bgr_mj = cv2.cvtColor(rgb_mj, cv2.COLOR_RGB2BGR)
                obs[f"observation.images.{camera_name}"] = bgr_mj

        obs["observation.keypoints"] = self.get_keypoint_positions().reshape(-1)

        return obs, {}

    def _clamp_target_matrix(self, current_mat, target_mat, duration):
        current_pos = current_mat[:3, 3]
        target_pos = target_mat[:3, 3]

        diff_vec = target_pos - current_pos
        dist = np.linalg.norm(diff_vec)
        max_dist = self.max_ee_linear_vel * duration

        if dist > max_dist:
            clamped_pos = current_pos + (diff_vec / dist) * max_dist
        else:
            clamped_pos = target_pos

        current_rot = current_mat[:3, :3]
        target_rot = target_mat[:3, :3]

        curr_quat = T.mat2quat(current_rot)
        targ_quat = T.mat2quat(target_rot)

        dot = np.abs(np.dot(curr_quat, targ_quat))
        dot = np.clip(dot, -1.0, 1.0)
        theta = 2 * np.arccos(dot)

        max_angle = self.max_ee_angular_vel * duration

        if theta > max_angle and theta > 1e-6:
            t = max_angle / theta
            dot_raw = np.dot(curr_quat, targ_quat)
            target_sign = np.sign(dot_raw)
            targ_quat_aligned = targ_quat * target_sign

            omega = np.arccos(np.clip(np.abs(dot_raw), -1, 1))
            if np.abs(omega) < 1e-6:
                clamped_quat = curr_quat
            else:
                sin_omega = np.sin(omega)
                w1 = np.sin((1 - t) * omega) / sin_omega
                w2 = np.sin(t * omega) / sin_omega
                clamped_quat = w1 * curr_quat + w2 * targ_quat_aligned

            clamped_rot = T.quat2mat(clamped_quat)
        else:
            clamped_rot = target_rot

        clamped_mat = np.eye(4)
        clamped_mat[:3, :3] = clamped_rot
        clamped_mat[:3, 3] = clamped_pos
        return clamped_mat

    def get_lerobot_features(self):
        features = {}

        # State: current EE pose (pos + rot6d) + gripper joint norm
        ee_pose_mat = self.get_ee_pose_mat()
        pos = ee_pose_mat[:3, 3]
        rotation_6D = ee_pose_mat[:2, :3].reshape(6)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        qpos_index = self.model.jnt_qposadr[joint_id]
        gripper_norm = np.array([self.data.qpos[qpos_index]])
        features["observation.state"] = np.concatenate([pos, rotation_6D, gripper_norm])

        # Action: FK on the commanded arm ctrl -> commanded EE pose, gripper
        # from ctrl[7]. Matches rekep_solver/env.py's convention so the action
        # field is the *commanded* target, not the snapshot of last_action.
        action_qpos = self.data.qpos[:13].copy()
        action_qpos[0:7] = self.data.ctrl[0:7]
        action_ee_mat = self.kin_helper.compute_fk_from_link_names(
            action_qpos, ["link_tcp"]
        )["link_tcp"]
        action_pos = action_ee_mat[:3, 3]
        action_rotation_6D = action_ee_mat[:2, :3].reshape(6)
        action_gripper_norm = np.array([self.data.ctrl[7]])
        features["action"] = np.concatenate([action_pos, action_rotation_6D, action_gripper_norm])

        for obj_name in self.object_map.keys():
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            obj_pos = self.data.xpos[obj_id]
            obj_quat = T.convert_quat(self.data.xquat[obj_id], to="xyzw")
            features[f"object.{obj_name}.pose"] = np.concatenate([obj_pos, obj_quat])

        features["observation.keypoints"] = self.get_keypoint_positions().reshape(-1)
        if self.episode_context is not None:
            features["observation.constraint_context"] = self.episode_context.copy()

        state_buf = self.save_state()
        features["state_buf"] = np.array(state_buf, dtype=np.float64)
        features["last_gripper_status"] = np.array([self.last_gripper_status], dtype=np.float64)

        return features

    def update_keypoints(self):
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

        for site_name in ['left_finger_site', 'right_finger_site']:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            pos = self.data.site_xpos[site_id]
            self.keypoints[idx] = {
                'obj_name': body_name,
                'site_name': site_name,
                'pos': pos
            }
            idx += 1

    def get_keypoint_positions(self):
        self.update_keypoints()
        return np.array([k['pos'] for k in self.keypoints.values()])

    def render(self):
        if self.render_mujoco:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    model=self.model,
                    data=self.data,
                )
            else:
                self.viewer.sync()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        self.model = None
        self.data = None

    def get_ee_pose_mat(self):
        link_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'link_tcp')
        link_ee_pos = self.data.site_xpos[link_ee_id]
        link_ee_R = self.data.site_xmat[link_ee_id].reshape(3, 3)
        transformation = np.eye(4)
        transformation[:3, :3] = link_ee_R
        transformation[:3, 3] = link_ee_pos
        return transformation

    def get_ee_pose_mj(self):
        link_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'link_tcp')
        link_ee_pos = self.data.site_xpos[link_ee_id]
        link_ee_quat = T.mat2quat(self.data.site_xmat[link_ee_id].reshape(3, 3))
        return np.concatenate([link_ee_pos, link_ee_quat])

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
        mask = self._full_state_mask()
        size = mujoco.mj_stateSize(self.model, mask)
        state_buf = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state_buf, mask)
        return state_buf

    def recover_state(self, state_buf, last_gripper_status) -> None:
        mask = self._full_state_mask()
        mujoco.mj_setState(self.model, self.data, state_buf, mask)
        self.last_gripper_status = last_gripper_status
        mujoco.mj_forward(self.model, self.data)
        self.update_keypoints()
        self._update_object_in_hand()

    def _update_object_in_hand(self):
        pass

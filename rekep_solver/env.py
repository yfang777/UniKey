from robot.cameras import CameraRenderers
from robot.model import resolve_robot_paths, scene_model_text
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import trimesh
import time
import yaml
import os

from scene_generator.utils.runtime_scene import runtime_scene_from_xml

from .ik_pinocchio_sapien import KinHelper

from . import transform_utils as T


class StepBudgetExceeded(Exception):
    """Raised inside ReKepEnv.step() when step_counter exceeds step_budget.

    Planner helpers convert this into an early-stop signal, finalize the
    current lerobot_features buffer, and return (stages, early_stopped=True).
    """
    pass


from .utils import (
    set_seed,
    bcolors,
    get_clock_time,
    angle_between_quats,
    angle_between_rotmat,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
    visualize_scene_and_sdf_plotly,
    _calc_cammatrices,
    get_body_site_map,
    get_object_map
)
from .domain_randomization import (
    DEFAULT_DOMAIN_RANDOMIZATION_CONFIG,
    DomainRandomizationConfig,
    build_domain_randomized_runtime_scene,
)



class ReKepEnv:
    def __init__(self, max_ee_linear_vel: float):
        """
        Initialize the XArm7 MuJoCo environment
        """
        with open("configurations/rekep_solver/config.yaml", "r") as f:
            self.cfg = yaml.safe_load(f)


        self.bounds_min = np.array(self.cfg['main']['bounds_min'])
        self.bounds_max = np.array(self.cfg['main']['bounds_max'])
        self.pointcloud_sample_num = np.array(self.cfg['main']['pointcloud_sample_num'])

        self.pointcloud_bounds_min = np.array(self.cfg['main']['pointcloud_bounds_min'])
        self.pointcloud_bounds_max = np.array(self.cfg['main']['pointcloud_bounds_max'])



        self.max_ee_linear_vel = max_ee_linear_vel
        self.max_ee_angular_vel = self.cfg['main']['max_ee_angular_vel']   # Max rotation: 2.0 rad/s


        self.interpolate_pos_step_size = self.cfg['main']['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.cfg['main']['interpolate_rot_step_size']

        self.render_width = self.cfg['main']['width']
        self.render_height = self.cfg['main']['height']
        self.cams = self.cfg['camera']['cam_name_list']
        # self.render_mujoco = self.cfg['visualizer']['render_mujoco']
        # self.save_video = self.cfg['visualizer']['save_video']
        self.video_dir = os.path.join(self.cfg['visualizer']['save_dir'], get_clock_time())
        self.fps = self.cfg['visualizer']['fps']
        self.fps_lerobot = self.cfg['lerobot']['fps']
        self.verbose = self.cfg["main"]["verbose"]

        self.dt = self.cfg['main']['timestep']

        self.min_depth = self.cfg['main']['min_depth']
        self.max_depth = self.cfg['main']['max_depth']

        # Gripper Action: open: 1, close: 0, no change: -1
        # Gripper Status: open: 1, close: 0
        self.gripper_open_action_value = 1
        self.gripper_close_action_value = 0
        self.gripper_null_action_value = -1

        # self.reset_joint_pos = np.array([0, 0, 0, 0, 0, 0, 0])
        self.reset_joint_pos = np.array([ 0., -0.78539816,  0., 0.52359878, 0., 1.30899694, 0.])

        self.control_hz = 50

        self.video_frames = {}
        for cam_name in self.cams:
            self.video_frames[cam_name] = []

        self.step_counter = 0
        self.step_budget = None  # set by planner helpers for early-stop
        self.viewer = None
        self.renderer = None



        set_seed(42)


    def reset(
        self,
        mujoco_xml_path,
        physic_parameter: dict | None = None,
        domain_randomization_seed: int | None = None,
        domain_randomization_output_dir: str | Path | None = None,
        domain_randomization_config: DomainRandomizationConfig = DEFAULT_DOMAIN_RANDOMIZATION_CONFIG,
        dummy_simulate_t: float = 0.0,
    ):
        """Reset the environment to initial state"""

        # (Re)set the scene
        if domain_randomization_seed is None:
            self.runtime_scene = runtime_scene_from_xml(mujoco_xml_path)
        else:
            if domain_randomization_output_dir is None:
                raise ValueError(
                    "domain_randomization_output_dir is required when "
                    "domain_randomization_seed is set."
                )
            self.runtime_scene = build_domain_randomized_runtime_scene(
                mujoco_xml_path,
                domain_randomization_output_dir,
                domain_randomization_seed,
                domain_randomization_config,
            )
        self.model = self.runtime_scene.model
        self.data = self.runtime_scene.data
        self.robot_variant = scene_model_text(self.model, "robot_variant")
        urdf_path, _ = resolve_robot_paths(self.robot_variant)
        self.kin_helper = KinHelper(urdf_path)
        self.cams = scene_model_text(self.model, "observation_cameras").split()
        # self.ik_solver = IKSolver()

        self.last_gripper_status = 1

        self.render_mujoco = False

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.renderer = CameraRenderers(self.model)

        # self.keypoint_helper = Keypoint(self.model, self.data, self.cfg)
        self.viewer = None
        self.site_map = get_body_site_map(mujoco_xml_path)
        self.object_map = get_object_map(mujoco_xml_path)

        # Set the value
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:7] = self.reset_joint_pos
        self.data.ctrl[0:7] = self.reset_joint_pos
        self.data.ctrl[7:9] = 0

        # self.data.qpos[0:13] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        mujoco.mj_forward(self.model, self.data)
        self.step_counter = 0
        self.step_budget = None
        self.in_hand_obj_name = None
        for cam_name in self.cams:
            self.video_frames[cam_name] = []
        self.lerobot_features = []

        self.object_in_hand_threshold = 5
        if physic_parameter is not None:
            if physic_parameter["object_in_hand_threshold"] is not None:
                self.object_in_hand_threshold = physic_parameter["object_in_hand_threshold"]

        if dummy_simulate_t < 0:
            raise ValueError("dummy_simulate_t must be >= 0")
        dummy_steps = int(round(dummy_simulate_t / self.dt))
        for _ in range(dummy_steps):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.step_counter = 0
        self.step_budget = None
        for cam_name in self.cams:
            self.video_frames[cam_name] = []
        self.lerobot_features = []

        if self.render_mujoco:
            self.render()
        self.update_keypoints()
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, "target_plate_keypoint")
        self.episode_context = None
        if target_id != -1:
            target = self.model.numeric_data[self.model.numeric_adr[target_id]]
            keypoints = self.get_keypoint_positions()
            self.episode_context = np.concatenate([[target], keypoints[[0, -2, -1]].reshape(-1)])
        elif "battery_1" in self.object_map:
            self.episode_context = self.get_keypoint_positions()[[0, -2, -1]].reshape(-1).copy()

        shoe_context_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, 'shoe_insert_lift')
        if shoe_context_id != -1:
            self.episode_context = self.get_keypoint_positions()[1:6].reshape(-1).copy()

        tray_context_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, 'shoe_tray_placement')
        if tray_context_id != -1:
            keypoints = self.get_keypoint_positions()
            gripper = keypoints[[0, -2, -1]].reshape(-1)
            self.episode_context = np.concatenate([keypoints[1:6].reshape(-1), gripper, gripper])


        return None

    def step(self, action, n=1):
        """
        Take a step in the environment

        Args:
            action: Control signal for the actuators 0, 1, 2, 3, 4, 5, 6, 7

        Returns:
            obs: Current observation
            reward: Reward for the step
            done: Whether the episode is done
            info: Additional information
        """
        # Set control signal
        assert action.shape == (9,)
        self.data.ctrl[0:9] = action

        # Step the simulation

        for _ in range(n):
            sim_time = self.data.time
            target_frame_time = len(self.lerobot_features) / self.fps_lerobot
            if sim_time >= target_frame_time:
                self.lerobot_features.append(self.get_lerobot_feature())

                if self.render_mujoco:
                    self.render()

            mujoco.mj_step(self.model, self.data)

            self.step_counter += 1
            if self.step_budget is not None and self.step_counter >= self.step_budget:
                self.last_frame_action = action
                raise StepBudgetExceeded(
                    f"step_counter ({self.step_counter}) reached step_budget "
                    f"({self.step_budget})"
                )

        self.last_frame_action = action



        # Get observation
        obs = None
        # self.get_cam_obs()

        # Calculate reward (placeholder)
        reward = 0.0

        # Check if done (placeholder)
        done = False

        # Additional info
        info = {}





        return obs, reward, done, info

    def add_optimization_time(self, time_in_seconds):
        pass

    def get_ee_pos(self):
        """Get the position of the end effector x, y, z"""
        link_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'link_tcp')
        link_ee_pos = self.data.site_xpos[link_ee_id]
        return link_ee_pos

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

    def get_arm_joint_postions(self):
        """Get the 7-DoF joint positions of the arm"""
        return self.data.qpos[0:7]

    def render(self):
        """Render the environment"""
        if self.render_mujoco:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    model=self.model,
                    data=self.data,
                )
            else:
                self.viewer.sync()

    def close(self):
        """Close the environment"""

        if self.viewer is not None:
            viewer = self.viewer
            self.viewer = None
            viewer.close()

        if hasattr(self, 'renderer') and self.renderer is not None:
            renderer = self.renderer
            self.renderer = None
            renderer.close()


    def get_cam_obs(self):
        """Get the camera observations"""
        cam_obs = dict()
        for cam_name in self.cams:
            renderer = self.renderer.for_camera(cam_name)
            # rgb
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            renderer.update_scene(self.data, camera=cam_name)
            rgb = renderer.render()


            # depth
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera=cam_name)
            depth = renderer.render()

            # points
            # Map Points into the world frame
            # get extrinsic and intrisic from _calc_cammatrices
            ex_mat, in_mat = _calc_cammatrices(self.data, cam_name, self.cfg['camera'])

            # Get image dimensions
            H, W = depth.shape

            # Create pixel coordinate grid
            u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            u = u.reshape(-1)
            v = v.reshape(-1)
            z = depth.reshape(-1)

            # Filter out invalid depth values (0 or negative)
            valid_mask = z > 0
            u = u[valid_mask]
            v = v[valid_mask]
            z = z[valid_mask]


            if len(z) > 0:
                # Extract camera intrinsics
                fx, fy = in_mat[0, 0], in_mat[1, 1]
                cx, cy = in_mat[0, 2], in_mat[1, 2]

                # Back-project to camera space
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                cam_points = np.stack([x, y, z], axis=1)  # shape: (N, 3)

                # Transform to world frame using extrinsics
                cam_points_homo = np.concatenate([cam_points, np.ones((cam_points.shape[0], 1))], axis=1)  # (N, 4)

                ex_mat_inv = np.linalg.inv(ex_mat)
                world_points = (ex_mat_inv @ cam_points_homo.T).T[:, :3]

                within_bounds_mask = (
                    (world_points[:, 0] >= self.bounds_min[0]) & (world_points[:, 0] <= self.bounds_max[0]) &
                    (world_points[:, 1] >= self.bounds_min[1]) & (world_points[:, 1] <= self.bounds_max[1]) &
                    (world_points[:, 2] >= self.bounds_min[2]) & (world_points[:, 2] <= self.bounds_max[2])
                )

                # Filter the points
                pcd = world_points[within_bounds_mask]

            else:
                pcd = np.empty((0, 3))

            # seg
            renderer.enable_segmentation_rendering()
            renderer.update_scene(self.data, camera=cam_name)
            seg = renderer.render()
            renderer.disable_segmentation_rendering()

            cam_obs[cam_name] = {
                'rgb': rgb,
                'depth': depth,
                'points': pcd,
                'seg': seg
            }
        return cam_obs

    def get_lerobot_feature(self):

        features = {}
        all_pcds = []
        for cam_name in self.cams:
            renderer = self.renderer.for_camera(cam_name)
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            renderer.update_scene(self.data, camera=cam_name)
            rgb = renderer.render()
            features[f"observation.images.{cam_name}"] = rgb

            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera=cam_name)
            depth = renderer.render()

            # Anything closer than 0.1m is 0; anything further than 2.0m is 255
            depth_norm = np.clip(depth, self.min_depth, self.max_depth)
            depth_norm = (depth_norm - self.min_depth) / (self.max_depth - self.min_depth)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_3ch = np.repeat(depth_uint8[:, :, np.newaxis], 3, axis=2)
            features[f"observation.images.{cam_name}_depth"] = depth_3ch

        #     # Pointcloud generation
        #     H, W = depth.shape
        #     u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        #     u = u.reshape(-1)
        #     v = v.reshape(-1)
        #     z = depth.reshape(-1)

        #     valid_mask = z > 0
        #     u = u[valid_mask]
        #     v = v[valid_mask]
        #     z = z[valid_mask]

        #     if len(z) > 0:
        #         ex_mat, in_mat = _calc_cammatrices(self.data, cam_name, self.cfg['camera'])
        #         fx, fy = in_mat[0, 0], in_mat[1, 1]
        #         cx, cy = in_mat[0, 2], in_mat[1, 2]

        #         x = (u - cx) * z / fx
        #         y = (v - cy) * z / fy
        #         cam_points = np.stack([x, y, z], axis=1)

        #         cam_points_homo = np.concatenate([cam_points, np.ones((cam_points.shape[0], 1))], axis=1)
        #         ex_mat_inv = np.linalg.inv(ex_mat)
        #         world_points = (ex_mat_inv @ cam_points_homo.T).T[:, :3]

        #         within_bounds_mask = (
        #             (world_points[:, 0] >= self.pointcloud_bounds_min[0]) & (world_points[:, 0] <= self.pointcloud_bounds_max[0]) &
        #             (world_points[:, 1] >= self.pointcloud_bounds_min[1]) & (world_points[:, 1] <= self.pointcloud_bounds_max[1]) &
        #             (world_points[:, 2] >= self.pointcloud_bounds_min[2]) & (world_points[:, 2] <= self.pointcloud_bounds_max[2])
        #         )
        #         all_pcds.append(world_points[within_bounds_mask])


        # if all_pcds:
        #     pcd = np.concatenate(all_pcds, axis=0)
        #     if len(pcd) > self.pointcloud_sample_num:
        #         # Random Sampling
        #         # indices = np.random.choice(len(pcd), self.pointcloud_sample_num, replace=False)
        #         # pcd = pcd[indices]

        #         # Farthest Point Sampling
        #         pcd_o3d = o3d.geometry.PointCloud()
        #         pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        #         pcd_down = pcd_o3d.farthest_point_down_sample(self.pointcloud_sample_num)
        #         pcd = np.asarray(pcd_down.points)

        #     elif len(pcd) < self.pointcloud_sample_num:
        #         padding = np.zeros((self.pointcloud_sample_num - len(pcd), 3))
        #         pcd = np.concatenate([pcd, padding], axis=0)
        # else:
        #     pcd = np.zeros((self.pointcloud_sample_num, 3))

        # features["observation.pointcloud"] = pcd.astype(np.float32)


        # State
        # Use Right Drive Joint as the gripper norm
        ee_pose_mat = self.get_ee_pose_mat()
        pos = ee_pose_mat[:3, 3]
        rotation_6D = ee_pose_mat[:2, :3].reshape(6)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        qpos_index = self.model.jnt_qposadr[joint_id]
        gripper_norm = np.array([self.data.qpos[qpos_index]])
        features["observation.state"] = np.concatenate([pos, rotation_6D, gripper_norm])               # shape: (10,)

        # Action: FK on commanded arm ctrl → commanded EE pose (numerically the
        # target_mat_clamped used by _move_to_waypoint's IK call), gripper from ctrl[7]
        action_qpos = self.data.qpos[:13].copy()
        action_qpos[0:7] = self.data.ctrl[0:7]
        action_ee_mat = self.kin_helper.compute_fk_from_link_names(
            action_qpos, ["link_tcp"]
        )["link_tcp"]
        action_pos = action_ee_mat[:3, 3]
        action_rotation_6D = action_ee_mat[:2, :3].reshape(6)
        action_gripper_norm = np.array([self.data.ctrl[7]])
        features["action"] = np.concatenate([action_pos, action_rotation_6D, action_gripper_norm])     # shape: (10,)

        # Object Info
        for obj_name in self.object_map.keys():
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            obj_pos = self.data.xpos[obj_id]
            obj_quat = T.convert_quat(self.data.xquat[obj_id], to="xyzw")
            features[f"object.{obj_name}.pose"] = np.concatenate([obj_pos, obj_quat])

        # observation
        features[f"observation.keypoints"] = self.get_keypoint_positions().reshape(-1)
        if self.episode_context is not None:
            features["observation.constraint_context"] = self.episode_context.copy()

        # Sim info
        state_buf, last_gripper_status = self.save_state()
        features["state_buf"] = np.array(state_buf, dtype=np.float64)

        features["last_gripper_status"] = np.array([self.last_gripper_status], dtype=np.float64)
        return features

    def add_last_frame_lerobot_features(self):
        self.lerobot_features.append(self.get_lerobot_feature())

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
                # original: 3.0
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

            # ======================================
            # = status and safety check
            # ======================================
            if np.any(target_pose[:3] < self.bounds_min) \
                 or np.any(target_pose[:3] > self.bounds_max):
                # print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
                target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

            # ======================================
            # = interpolation
            # ======================================
            current_pose = self.get_ee_pose()
            pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
            rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
            pos_is_close = pos_diff < self.interpolate_pos_step_size
            rot_is_close = rot_diff < self.interpolate_rot_step_size
            if pos_is_close and rot_is_close:
                # self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
                pose_seq = np.array([target_pose])
            else:
                num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
                pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
                # self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')


            ### move to target pose
            total_steps = 0
            for pose in pose_seq[:-1]:
                steps = self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
                total_steps += steps
            # move to the final pose with required precision
            pose = pose_seq[-1]
            steps = self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=max_steps)
            total_steps += steps

            # compute error
            pos_error, rot_error = self.compute_target_delta_ee(target_pose)
            # print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

            ### apply gripper action
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
        self._update_object_in_hand()

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
        self._update_object_in_hand()


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

            # 4. Convert to Matrix for your IK Solver
            target_mat_clamped = T.convert_pose_quat2mat(clamped_target_pose)

            # 5. Solve IK for the CLAMPED target
            qpos, success, error = self.kin_helper.compute_ik_from_mat(self.data.qpos[:13], target_mat_clamped)
            if self.last_gripper_status == 1.0:
                self.step(np.concatenate([qpos[0:7], [0, 0]]), n=steps_n)
            else:
                self.step(np.concatenate([qpos[0:7], [0.85, 0.85]]), n=steps_n)


            # qpos, success, error = self.kin_helper.compute_ik_from_mat(self.data.qpos[:13], target_mat_world)
            # target_joints = qpos[0:7]
            # steps_n = int(1 / self.dt * 1 / self.control_hz)
            # step_duration = steps_n * self.dt
            # safe_joints = self._apply_joint_velocity_limit(target_joints, step_duration)
            # if self.last_gripper_status == 1.0:
            #     action = np.concatenate([safe_joints, [0, 0]])
            #     self.step(action, n=steps_n)
            # else:
            #     action = np.concatenate([safe_joints, [0.85, 0.85]])
            #     self.step(action, n=steps_n)



            # qpos, success, error = self.kin_helper.compute_ik_from_mat(self.data.qpos[:13], target_mat_world)
            # if self.last_gripper_status == 1.0:
            #     self.step(np.concatenate([qpos[0:7], [0, 0]]), n= int(1 /self.dt * 1 / self.control_hz))
            # else:
            #     self.step(np.concatenate([qpos[0:7], [0.85, 0.85]]), n= int(1 /self.dt * 1 / self.control_hz))
            self.update_keypoints()
            self._update_object_in_hand()

            count += 1
        if count == max_steps:
            print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose not reached after {max_steps} steps (pos_error: {pos_errors[-1].round(5)}, rot_error: {np.rad2deg(rot_errors[-1]).round(4)}){bcolors.ENDC}')

        # print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose not reached after {max_steps} steps (pos_error: {pos_errors[-1].round(4)}, rot_error: {np.rad2deg(rot_errors[-1]).round(4)}){bcolors.ENDC}')
        return count

    def _clamp_ee_target(self, current_pose, target_pose, duration):
        """
        Calculates a new target pose that is within the velocity limits
        from the current pose given the time duration.
        """
        # --- Linear Clamping ---
        current_pos = current_pose[:3]
        target_pos = target_pose[:3]

        direction = target_pos - current_pos
        dist = np.linalg.norm(direction)

        max_dist = self.max_ee_linear_vel * duration

        if dist > max_dist:
            # Scale the target back to the limit
            clamped_pos = current_pos + (direction / dist) * max_dist
        else:
            clamped_pos = target_pos

        # --- Rotational Clamping (SLERP) ---
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
        # calculate position delta
        pos_diff = (target_pos - current_pos).flatten()
        pos_error = np.linalg.norm(pos_diff)
        # calculate rotation delta
        rot_error = angle_between_rotmat(current_rotmat, target_rotmat)
        # print status
        # self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Curr pose: {current_pos}, {current_xyzw} (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
        # self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Goal pose: {target_pos}, {target_xyzw} (pos_thres: {pos_threshold}, rot_thres: {rot_threshold}){bcolors.ENDC}')

        if pos_error < pos_threshold and rot_error < np.deg2rad(rot_threshold):
            # self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose reached (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
            return True, pos_error, rot_error


        return False, pos_error, rot_error

    def compute_target_delta_ee(self, target_pose):
        target_pos, target_xyzw = target_pose[:3], target_pose[3:]
        ee_pose = self.get_ee_pose()
        ee_pos, ee_xyzw = ee_pose[:3], ee_pose[3:]
        pos_diff = np.linalg.norm(ee_pos - target_pos)
        rot_diff = angle_between_quats(ee_xyzw, target_xyzw)


        return pos_diff, rot_diff

    # ======================================
    # = SDF Voxels / Collision Points
    # ======================================
    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        """
        open3d-based SDF computation
        1. recursively get all usd prim and get their vertices and faces
        2. compute SDF using open3d
        """
        import open3d as o3d


        # 'link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'xarm_gripper_base_link', 'left_outer_knuckle', 'left_finger', 'left_inner_knuckle', 'right_outer_knuckle', 'right_finger', 'right_inner_knuckle']
        # ['xarm_gripper_ba se_link', 'left_outer_knuckle', 'left_finger', 'left_inner_knuckle', 'right_outer_knuckle', 'right_finger', 'right_inner_knuckle']


        start = time.time()
        exclude_names = []
        if exclude_obj_in_hand:
            if self.in_hand_obj_name is not None:
                exclude_names.append(self.in_hand_obj_name)
        trimesh_objects = []

        for obj_name in self.object_map.keys():
            if any([name in obj_name for name in exclude_names]):
                continue

            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            obj_pos = self.data.xpos[obj_id]

            # obj_init_quat = T.convert_quat(np.array(self.object_map[obj_name]["initial_quat"]), to="xyzw")
            # obj_quat = T.convert_quat(self.data.xquat[obj_id], to="xyzw")
            # obj_quat_final = T.convert_quat(T.quat_multiply(obj_quat, obj_init_quat), to="wxyz")

            obj_quat_final = self.data.xquat[obj_id]

            obj_pose_transform = trimesh.transformations.quaternion_matrix(obj_quat_final)
            obj_pose_transform[:3, 3] = obj_pos

            obj_scale_transform = np.array(self.object_map[obj_name]["scale"])

            obj_mesh = trimesh.load(self.object_map[obj_name]['mesh_file'])
            obj_mesh.apply_scale(obj_scale_transform)
            obj_mesh.apply_transform(obj_pose_transform)


            trimesh_objects.append(obj_mesh)

        plane_size = 3  # adjust to cover your workspace bounds
        plane_mesh = trimesh.creation.box(
            extents=[plane_size, plane_size, 0.001]  # very thin in z
        )
        plane_mesh.apply_translation([0, 0, -0.07])
        trimesh_objects.append(plane_mesh)

        cylinder_radius = 0.1
        cylinder_height = 0.3
        cylinder_mesh = trimesh.creation.cylinder(
            radius=cylinder_radius,
            height=cylinder_height,
            sections=32  # smoother cylinder
        )
        cylinder_mesh.apply_translation([0, 0, 0.3])  # match plane z
        trimesh_objects.append(cylinder_mesh)


        # chain trimesh objects
        scene_mesh = trimesh.util.concatenate(trimesh_objects)
        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        vertex_positions = scene_mesh.vertices
        triangle_indices = scene_mesh.faces
        vertex_positions = o3d.core.Tensor(vertex_positions, dtype=o3d.core.Dtype.Float32)
        triangle_indices = o3d.core.Tensor(triangle_indices, dtype=o3d.core.Dtype.UInt32)
        _ = scene.add_triangles(vertex_positions, triangle_indices)  # we do not need the geometry ID for mesh
        # create a grid
        shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
        steps = (self.bounds_max - self.bounds_min) / shape

        # do not include the last point
        grid = np.mgrid[
            self.bounds_min[0]:self.bounds_max[0]:complex(shape[0]),
            self.bounds_min[1]:self.bounds_max[1]:complex(shape[1]),
            self.bounds_min[2]:self.bounds_max[2]:complex(shape[2])
        ]


        grid = grid.reshape(3, -1).T
        # compute SDF
        sdf_voxels = scene.compute_signed_distance(grid.astype(np.float32))
        # convert back to np array
        sdf_voxels = sdf_voxels.cpu().numpy()
        # open3d has flipped sign from our convention
        sdf_voxels = -sdf_voxels
        sdf_voxels = sdf_voxels.reshape(shape)
        # self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] SDF voxels computed in {time.time() - start:.4f} seconds{bcolors.ENDC}')


        return sdf_voxels

    def sample_scene_points(self, num_points=1000):
        '''
        Returns:
            np.ndarray: scene points in the world frame of shape (N, 3)
        '''
        scene_points = []

        # add object in hand collision points
        for obj_name in self.object_map.keys():
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            obj_pos = self.data.xpos[obj_id]

            # obj_init_quat = T.convert_quat(np.array(self.object_map[obj_name]["initial_quat"]), to="xyzw")
            # obj_quat = T.convert_quat(self.data.xquat[obj_id], to="xyzw")

            # obj_quat_final = T.convert_quat(T.quat_multiply(obj_quat, obj_init_quat), to="wxyz")

            obj_quat_final = self.data.xquat[obj_id]
            obj_pose_transform = trimesh.transformations.quaternion_matrix(obj_quat_final)
            obj_pose_transform[:3, 3] = obj_pos

            obj_scale_transform = np.array(self.object_map[obj_name]["scale"])

            obj_mesh = trimesh.load(self.object_map[obj_name]['mesh_file'])
            obj_mesh.apply_scale(obj_scale_transform)
            obj_mesh.apply_transform(obj_pose_transform)
            points_transformed = obj_mesh.sample(num_points)
            scene_points.append(points_transformed)

        link_names = ['link7', 'xarm_gripper_base_link', 'left_outer_knuckle', 'left_finger', 'left_inner_knuckle', 'right_outer_knuckle', 'right_finger', 'right_inner_knuckle']

        pcd = self.kin_helper.compute_robot_pcd(self.data.qpos[:13], link_names=['link_base', 'link1'], num_pts=None, pcd_name="xarm7")
        scene_points.append(pcd)


        # Add cylinder mesh points
        cylinder_radius = 0.1
        cylinder_height = 0.3
        cylinder_mesh = trimesh.creation.cylinder(
            radius=cylinder_radius,
            height=cylinder_height,
            sections=32
        )
        cylinder_mesh.apply_translation([0, 0, 0.3])
        scene_points.append(cylinder_mesh.sample(num_points))

        scene_points = np.concatenate(scene_points, axis=0)


        return scene_points


    def get_collision_points(self, noise=True):
        """
        Get the points of the gripper and any object in hand.
        """


        collision_points = []
        # add object in hand collision points
        if self.in_hand_obj_name is not None:
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.in_hand_obj_name)
            obj_pos = self.data.xpos[obj_id]

            obj_quat_final = self.data.xquat[obj_id]
            obj_pose_transform = trimesh.transformations.quaternion_matrix(obj_quat_final)
            obj_pose_transform[:3, 3] = obj_pos

            obj_scale_transform = np.array(self.object_map[self.in_hand_obj_name]["scale"])

            obj_mesh = trimesh.load(self.object_map[self.in_hand_obj_name]['mesh_file'])
            obj_mesh.apply_scale(obj_scale_transform)
            obj_mesh.apply_transform(obj_pose_transform)
            points_transformed = obj_mesh.sample(1000)
            collision_points.append(points_transformed)

        # add gripper and arm collision points
        link_names = ['xarm_gripper_base_link', 'left_outer_knuckle', 'left_finger', 'left_inner_knuckle', 'right_outer_knuckle', 'right_finger', 'right_inner_knuckle', 'link7', 'link6']
        pcd = self.kin_helper.compute_robot_pcd(self.data.qpos[:13], link_names=link_names, num_pts=None, pcd_name="xarm7")
        collision_points.append(pcd)

        collision_points = np.concatenate(collision_points, axis=0)
        return collision_points

    def _update_object_in_hand(self):
        """
        Update the object in hand.
        """

        right_finger_touch = np.linalg.norm(self.data.sensor('right_finger_touch').data)
        left_finger_touch = np.linalg.norm(self.data.sensor('left_finger_touch').data)



        in_touch = right_finger_touch > self.object_in_hand_threshold or left_finger_touch > self.object_in_hand_threshold

        link_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'link_tcp')
        link_ee_pos = self.data.site_xpos[link_ee_id]
        for keypoint_idx, keypoint_info in self.keypoints.items():
            site_name = self.keypoints[keypoint_idx]["site_name"]
            if site_name in ['link_tcp', 'left_finger_site', 'right_finger_site']:
                continue
            pos = self.keypoints[keypoint_idx]["pos"]
            if np.linalg.norm(link_ee_pos - pos) < 0.03 and in_touch:
                obj_name = self.get_object_by_keypoint(keypoint_idx)
                self.in_hand_obj_name = obj_name
                return

        self.in_hand_obj_name = None


    # ======================================
    # = Keypoints
    # ======================================


    def register_keypoints(self):
        pass

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


    def get_keypoint_positions(self):
        """
        Returns keypoint positions from all registered keypoints.
        Shape: (N, 3)
        """
        self.update_keypoints()


        return np.array([k['pos'] for k in self.keypoints.values()])

    def get_object_by_keypoint(self, keypoint_idx):
        """
        Args:
            keypoint_idx (int): the index of the keypoint
        Returns:
            pointer: the object that the keypoint is associated with
        Given the keypoint index, this function returns the name of the object that the keypoint is associated with.
        """
        return self.keypoints[keypoint_idx]['obj_name']

    def is_grasping(self, candidate_obj):
        '''
        Args:
            candidate_obj (str): the name of the object
        Returns:
            bool: True if the object is grasped, False otherwise
        '''
        return self.in_hand_obj_name == candidate_obj

    def is_grasping_keypoint(self, keypoint_idx):
        '''
        Args:
            keypoint (str): the name of the object
        Returns:
            bool: True if the object is grasped, False otherwise
        '''

        right_finger_touch = np.linalg.norm(self.data.sensor('right_finger_touch').data)
        left_finger_touch = np.linalg.norm(self.data.sensor('left_finger_touch').data)


        in_touch = right_finger_touch > 10 or left_finger_touch > 10

        link_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'link_tcp')
        link_ee_pos = self.data.site_xpos[link_ee_id]
        pos = self.keypoints[keypoint_idx]["pos"]
        if np.linalg.norm(link_ee_pos - pos) < 0.03 and in_touch:
            return True
        else:
            return False

    def get_object_pose(self, obj_name):
        """
        Args:
            obj_name (str): the name of the object
        Returns:
            np.ndarray: the pose of the object in the world frame
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        obj_pos = self.data.xpos[obj_id]
        obj_mat = self.data.xmat[obj_id].reshape(3, 3)

        if obj_name == "039_mug_1":
            # +z: 90 degree
            # grasp2obj = np.array([
            #     [0, -1, 0],
            #     [1,  0, 0],
            #     [0,  0, 1]
            # ])

            grasp2obj = np.array([
                [-1, 0, 0],
                [0,  -1, 0],
                [0,  0, 1]
            ])


            # grasp2obj = np.eye(3)

        else:
            grasp2obj = np.eye(3)

        world2gripper = np.array([[0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0]])

        gripper_target_mat = grasp2obj @ obj_mat @ world2gripper

        gripper_target_quat = T.mat2quat(gripper_target_mat)



        obj_pose = np.concatenate([obj_pos, gripper_target_quat])

        return obj_pose


    @staticmethod
    def _full_state_mask():
        s = mujoco.mjtState
        # Compose “ALL” explicitly (works across MuJoCo versions)
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


    def recover_state(self, state_buf, last_gripper_status) -> None:
        """
        Restore a state produced by save_state().
        """
        mask = self._full_state_mask()
        state_buf = np.array(state_buf, dtype=np.float64).reshape(-1, 1)

        mujoco.mj_setState(self.model, self.data, state_buf, mask)

        self.last_gripper_status = last_gripper_status[0]
        self.data.time = 0
        # Recompute all derived quantities (sites, sensors, contacts, etc.)
        mujoco.mj_forward(self.model, self.data)
        self.update_keypoints()
        self._update_object_in_hand()
        self.render()
        self.lerobot_features = []
        # Each planner-augment run gets a fresh time budget; the previous
        # rollout's elapsed mujoco steps do not count against it.
        self.step_counter = 0
        self.step_budget = None

    def get_imgs_with_keypoints(self):
        import cv2

        imgs = []
        for cam_name in self.cams:
            mujoco_renderer = self.renderer.for_camera(cam_name)
            # rgb
            mujoco_renderer.disable_depth_rendering()
            mujoco_renderer.disable_segmentation_rendering()
            mujoco_renderer.update_scene(self.data, camera=cam_name)
            rgb = mujoco_renderer.render()


            # Extrinsic And Intrinsic
            ex_mat, in_mat = _calc_cammatrices(self.data, cam_name, self.cfg['camera'])

            # Convert keypoints to homogeneous coordinates
            keypoints_3d_world = self.get_keypoint_positions()
            num_points = keypoints_3d_world.shape[0]
            homog_points = np.hstack([keypoints_3d_world, np.ones((num_points, 1))])
            cam_points = (ex_mat @ homog_points.T).T[:, :3]

            # Project to 2D using intrinsic matrix
            pixels = []
            for point in cam_points:
                if point[2] <= 1e-6:
                    pixels.append([-1, -1])  # behind the camera or at zero depth
                    continue
                proj = in_mat[:3, :3] @ (point / point[2])
                pixels.append(proj[:2])

            for id, (u, v) in enumerate(pixels):
                if u >= 0 and v >= 0:
                    text = str(id)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness = 1

                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    # Calculate position to center the text in the circle
                    text_x = int(u - text_width / 2)
                    text_y = int(v + text_height / 2)  # Vertical centering

                    # Draw the circle
                    cv2.circle(rgb, (int(u), int(v)), radius=6, color=(0, 0, 255), thickness=-1)

                    # Draw the centered text
                    cv2.putText(rgb, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            imgs.append(rgb)
            # cam_obs["cam_name"] = rgb
        return imgs

    def teleop_control(self):
        """
        Teleoperate the robot using keyboard input (no Enter required).
        Controls:
            - w/s: +z/-z
            - a/d: -y/+y
            - q/e: -x/+x
            - i/k: rotate +x/-x (roll)
            - j/l: rotate +y/-y (pitch)
            - n/m: rotate +z/-z (yaw)
            - o: open gripper
            - c: close gripper
            - x: exit teleop mode
        """
        import sys
        import termios
        import tty
        import select

        if not sys.stdin.isatty():
            raise RuntimeError("teleoperation requires an interactive terminal")

        def getch():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                [i, _, _] = select.select([fd], [], [], 0.1)
                if i:
                    ch = sys.stdin.read(1)
                else:
                    ch = ''
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        print("--- Teleoperation Mode ---")
        print("Controls:")
        print("  q/e: +z/-z, a/d: -y/+y,w/s: -x/+x")
        print("  i/k: roll +x/-x, j/l: pitch +y/-y, n/m: yaw +z/-z")
        print("  o: open gripper, c: close gripper, x: exit")
        print("Press keys to control. Press 'x' to exit.")

        pos_step = 0.01
        rot_step_deg = 5
        rot_step = np.deg2rad(rot_step_deg)

        current_pose = self.get_ee_pose().copy()
        gripper_action = self.gripper_null_action_value

        while True:
            # print(f"Current pose: pos={current_pose[:3].round(3)}, quat={current_pose[3:].round(3)}")
            key = getch().lower()
            if not key:
                continue
            if key == 'x':
                print("Exiting teleop mode.")
                break
            elif key == 'q':
                current_pose[2] += pos_step
            elif key == 'e':
                current_pose[2] -= pos_step
            elif key == 'a':
                current_pose[1] -= pos_step
            elif key == 'd':
                current_pose[1] += pos_step
            elif key == 's':
                current_pose[0] -= pos_step
            elif key == 'w':
                current_pose[0] += pos_step
            elif key == 'i':
                # roll +x
                axis = np.array([1, 0, 0]) * rot_step
                dq = T.axisangle2quat(axis)
                current_pose[3:] = T.quat_multiply(dq, current_pose[3:])
            elif key == 'k':
                # roll -x
                axis = np.array([1, 0, 0]) * (-rot_step)
                dq = T.axisangle2quat(axis)
                current_pose[3:] = T.quat_multiply(dq, current_pose[3:])
            elif key == 'j':
                # pitch +y
                axis = np.array([0, 1, 0]) * rot_step
                dq = T.axisangle2quat(axis)
                current_pose[3:] = T.quat_multiply(dq, current_pose[3:])
            elif key == 'l':
                # pitch -y
                axis = np.array([0, 1, 0]) * (-rot_step)
                dq = T.axisangle2quat(axis)
                current_pose[3:] = T.quat_multiply(dq, current_pose[3:])
            elif key == 'n':
                # yaw +z
                axis = np.array([0, 0, 1]) * rot_step
                dq = T.axisangle2quat(axis)
                current_pose[3:] = T.quat_multiply(dq, current_pose[3:])
            elif key == 'm':
                # yaw -z
                axis = np.array([0, 0, 1]) * (-rot_step)
                dq = T.axisangle2quat(axis)
                current_pose[3:] = T.quat_multiply(dq, current_pose[3:])
            elif key == 'o':
                self.open_gripper()
                print("in_hand_obj_name:", self.in_hand_obj_name)
                continue
            elif key == 'c':
                self.close_gripper()
                print("in_hand_obj_name:", self.in_hand_obj_name)
                continue
            else:
                print(f"Unknown command: {key}")
                continue

            print("in_hand_obj_name:", self.in_hand_obj_name)
            self._move_to_waypoint(current_pose[:7], pos_threshold=0.003, rot_threshold=1)

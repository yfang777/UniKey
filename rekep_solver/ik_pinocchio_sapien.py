from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np

import pinocchio as pin
import transforms3d


KINEMATICS_URDF_TAGS = {"origin", "parent", "child", "axis", "limit"}
SAPIEN_QPOS_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "drive_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]


def build_kinematics_urdf(urdf_path: str) -> str:
    """Return a kinematics-only URDF string for Pinocchio."""
    root = ET.parse(urdf_path).getroot()
    robot = ET.Element("robot", {"name": root.attrib["name"]})

    for elem in root:
        if elem.tag == "link":
            robot.append(ET.Element("link", {"name": elem.attrib["name"]}))
        elif elem.tag == "joint":
            joint = ET.Element("joint", elem.attrib)
            for child in elem:
                if child.tag in KINEMATICS_URDF_TAGS:
                    joint.append(copy.deepcopy(child))
            robot.append(joint)

    return ET.tostring(robot, encoding="unicode")


class PinocchioPose:
    def __init__(self, transform: np.ndarray):
        self._transform = transform

    def to_transformation_matrix(self) -> np.ndarray:
        return self._transform.copy()


class SapienStylePinocchioModel:
    """Direct Pinocchio model that mirrors SAPIEN's PinocchioModel wrapper."""

    def __init__(
        self,
        urdf_path: str,
        joint_names: list[str],
        link_names: list[str],
    ):
        self.model = pin.buildModelFromXML(build_kinematics_urdf(urdf_path))
        self.data = self.model.createData()
        self.set_joint_order(joint_names)
        self.set_link_order(link_names)

    def set_joint_order(self, names: list[str]) -> None:
        index_s2p = np.zeros(self.model.nv, dtype=np.int64)
        count = 0
        for name in names:
            joint_id = self.model.getJointId(name)
            for joint_offset in range(self.model.nvs[joint_id]):
                index_s2p[count] = self.model.idx_vs[joint_id] + joint_offset
                count += 1

        self.index_s2p = index_s2p
        self.index_p2s = np.zeros_like(index_s2p)
        for sapien_index, pin_index in enumerate(index_s2p):
            self.index_p2s[pin_index] = sapien_index

        self.QIDX = np.zeros(len(names), dtype=np.int64)
        self.NQ = np.zeros(len(names), dtype=np.int64)
        self.NV = np.zeros(len(names), dtype=np.int64)
        for index, name in enumerate(names):
            joint_id = self.model.getJointId(name)
            self.QIDX[index] = self.model.idx_qs[joint_id]
            self.NQ[index] = self.model.nqs[joint_id]
            self.NV[index] = self.model.nvs[joint_id]

    def set_link_order(self, names: list[str]) -> None:
        self.link_id_to_frame_index = np.array(
            [self.model.getFrameId(name, pin.BODY) for name in names],
            dtype=np.int64,
        )

    def q_p2s(self, q_pin: np.ndarray) -> np.ndarray:
        q_sapien = np.zeros(self.model.nv)
        count = 0
        for index in range(len(self.QIDX)):
            q_index = self.QIDX[index]
            if self.NQ[index] == 1:
                q_sapien[count] = q_pin[q_index]
            elif self.NQ[index] == 2:
                q_sapien[count] = np.arctan2(q_pin[q_index + 1], q_pin[q_index])
            else:
                raise ValueError(
                    "Unsupported joint in computation. Currently support: fixed, revolute, prismatic"
                )
            count += self.NV[index]

        assert count == len(self.NV)
        return q_sapien

    def q_s2p(self, q_sapien: np.ndarray) -> np.ndarray:
        q_pin = np.zeros(self.model.nq)
        count = 0
        for index in range(len(self.QIDX)):
            q_index = self.QIDX[index]
            if self.NQ[index] == 1:
                q_pin[q_index] = q_sapien[count]
            elif self.NQ[index] == 2:
                q_pin[q_index] = np.cos(q_sapien[count])
                q_pin[q_index + 1] = np.sin(q_sapien[count])
            else:
                raise ValueError(
                    "Unsupported joint in computation. Currently support: fixed, revolute, prismatic"
                )
            count += self.NV[index]

        assert count == len(self.NV)
        return q_pin

    def compute_forward_kinematics(self, qpos: np.ndarray) -> None:
        pin.forwardKinematics(self.model, self.data, self.q_s2p(qpos))

    def get_link_pose(self, index: int) -> PinocchioPose:
        frame_id = int(self.link_id_to_frame_index[index])
        frame = self.model.frames[frame_id]
        link_to_world = self.data.oMi[frame.parentJoint] * frame.placement
        transform = np.eye(4)
        transform[:3, :3] = link_to_world.rotation
        transform[:3, 3] = link_to_world.translation
        return PinocchioPose(transform)

    def compute_full_jacobian(self, qpos: np.ndarray) -> None:
        pin.computeJointJacobians(self.model, self.data, self.q_s2p(qpos))

    def get_link_jacobian(self, index: int, local: bool = False) -> np.ndarray:
        frame_id = int(self.link_id_to_frame_index[index])
        frame = self.model.frames[frame_id]
        link_to_world = self.data.oMi[frame.parentJoint] * frame.placement
        jacobian = pin.getJointJacobian(
            self.model,
            self.data,
            frame.parentJoint,
            pin.ReferenceFrame.WORLD,
        )
        if local:
            jacobian = link_to_world.toActionMatrixInverse() @ jacobian
        return jacobian[:, self.index_s2p]

    @staticmethod
    def _pose_to_transform(pose) -> np.ndarray:
        if hasattr(pose, "to_transformation_matrix"):
            return pose.to_transformation_matrix()
        return np.asarray(pose, dtype=np.float64)

    def compute_inverse_kinematics(
        self,
        link_index: int,
        pose,
        initial_qpos: Optional[np.ndarray] = None,
        active_qmask: Optional[np.ndarray] = None,
        eps: float = 1e-4,
        max_iterations: int = 1000,
        dt: float = 0.1,
        damp: float = 1e-6,
    ) -> tuple[np.ndarray, bool, float]:
        if initial_qpos is None:
            q_pin = pin.neutral(self.model)
        else:
            q_pin = self.q_s2p(initial_qpos)

        if active_qmask is None:
            mask = np.ones(self.model.nv)
        else:
            mask = np.array(active_qmask)[self.index_p2s]
        mask = np.diag(mask)

        frame_id = int(self.link_id_to_frame_index[link_index])
        frame = self.model.frames[frame_id]
        joint_id = frame.parentJoint

        transform = self._pose_to_transform(pose)
        link_to_world_desired = pin.SE3()
        link_to_world_desired.translation[:] = transform[:3, 3]
        link_to_world_desired.rotation[:] = transform[:3, :3]
        joint_to_world_desired = link_to_world_desired * frame.placement.inverse()

        best_error = 1e10
        best_q_pin = np.array(q_pin)
        success = False

        for _ in range(max_iterations):
            pin.forwardKinematics(self.model, self.data, q_pin)
            joint_to_target = self.data.oMi[joint_id].actInv(joint_to_world_desired)
            error_vec = pin.log6(joint_to_target).vector
            error = float(np.linalg.norm(error_vec))
            if error < best_error:
                best_error = error
                best_q_pin = q_pin

            if error < eps:
                success = True
                break

            jacobian = pin.computeJointJacobian(self.model, self.data, q_pin, joint_id)
            jacobian_log = pin.Jlog6(joint_to_target.inverse())
            jacobian = -jacobian_log @ jacobian
            jacobian = jacobian @ mask

            jj_t = jacobian @ jacobian.T
            jj_t[np.diag_indices_from(jj_t)] += damp
            velocity = -(jacobian.T @ np.linalg.solve(jj_t, error_vec))
            q_pin = pin.integrate(self.model, q_pin, velocity * dt)

        return self.q_p2s(best_q_pin), success, best_error


def trimesh_to_open3d(tri_mesh):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    if tri_mesh.visual.kind == "vertex":
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            tri_mesh.visual.vertex_colors[:, :3] / 255.0
        )
    elif tri_mesh.visual.kind == "face":
        mesh.triangle_colors = o3d.utility.Vector3dVector(
            tri_mesh.visual.face_colors[:, :3] / 255.0
        )
    return mesh


class KinHelper:
    """Pinocchio-only kinematics helper using SAPIEN-wrapper numerical conventions."""

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = str(urdf_path)
        self.eef_name = "link_tcp"
        self.robot_name = "xarm7_gripper"
        self.active_qmask = np.array([True] * 7 + [False] * 6)

        self.link_names = [
            link.attrib["name"]
            for link in ET.parse(self.urdf_path).getroot().findall("link")
        ]
        self.robot_model = SapienStylePinocchioModel(
            self.urdf_path,
            SAPIEN_QPOS_JOINT_NAMES,
            self.link_names,
        )
        self.link_name_to_idx = {
            link_name: link_idx for link_idx, link_name in enumerate(self.link_names)
        }
        self.sapien_eef_idx = self.link_name_to_idx[self.eef_name]

        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        self.robot_meshes_loaded = False
        self.pcd_dict = {}
        self.tool_meshes = {}

    def _load_robot_meshes(self) -> None:
        if self.robot_meshes_loaded:
            return

        import trimesh
        from yourdfpy import URDF

        urdf_robot = URDF.load(self.urdf_path)
        for link_name, link in urdf_robot.link_map.items():
            if len(link.collisions) > 0:
                collision = link.collisions[0]
                if collision.geometry.mesh is not None:
                    mesh_path = urdf_robot._filename_handler(
                        collision.geometry.mesh.filename
                    )
                    mesh = trimesh.load(mesh_path)
                    self.meshes[link.name] = trimesh_to_open3d(mesh)
                    self.meshes[link.name].compute_vertex_normals()
                    self.meshes[link.name].paint_uniform_color([0.2, 0.2, 0.2])
                    self.scales[link.name] = (
                        collision.geometry.mesh.scale
                        if collision.geometry.mesh.scale is not None
                        else 1.0
                    )
                    self.offsets[link.name] = collision.origin
        self.robot_meshes_loaded = True

    def compute_fk_from_link_names(
        self,
        qpos: np.ndarray,
        link_names: list[str],
        in_obj_frame: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics of robot links given joint positions."""
        self.robot_model.compute_forward_kinematics(qpos)
        link_idx_ls = [self.link_name_to_idx[link_name] for link_name in link_names]
        poses_ls = self.compute_fk_from_link_idx(qpos, link_idx_ls)
        if in_obj_frame:
            self._load_robot_meshes()
            for index, link_name in enumerate(link_names):
                if link_name in self.offsets:
                    poses_ls[index] = poses_ls[index] @ self.offsets[link_name]
        return {link_name: pose for link_name, pose in zip(link_names, poses_ls)}

    def compute_all_fk(
        self,
        qpos: np.ndarray,
        in_obj_frame: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics of all robot links given joint positions."""
        return self.compute_fk_from_link_names(qpos, self.link_names, in_obj_frame)

    def compute_ik_euler(
        self,
        initial_qpos: np.ndarray,
        cartesian: np.ndarray,
        damp: float = 1e-1,
    ) -> tuple[np.ndarray, bool, float]:
        """Compute inverse kinematics given initial joint pos and target pose."""
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = transforms3d.euler.euler2mat(
            ai=cartesian[3],
            aj=cartesian[4],
            ak=cartesian[5],
            axes="sxyz",
        )
        tf_mat[:3, 3] = cartesian[0:3]
        return self.compute_ik_from_mat(initial_qpos, tf_mat, damp=damp)

    def compute_ik_from_mat(
        self,
        initial_qpos: np.ndarray,
        tf_mat: np.ndarray,
        damp: float = 1e-1,
    ) -> tuple[np.ndarray, bool, float]:
        """Compute IK given initial joint pos and target pose in matrix form."""
        return self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx,
            pose=tf_mat,
            initial_qpos=initial_qpos,
            active_qmask=self.active_qmask,
            eps=1e-3,
            damp=damp,
        )

    def _mesh_poses_to_pc(
        self,
        poses: np.ndarray,
        meshes: list,
        offsets: list[np.ndarray],
        num_pts: list[int],
        scales: list[int],
        pcd_name: Optional[str] = None,
    ) -> np.ndarray:
        assert poses.shape[0] == len(meshes)
        assert poses.shape[0] == len(offsets)
        assert poses.shape[0] == len(num_pts)
        assert poses.shape[0] == len(scales)

        all_pc = []
        for index in range(poses.shape[0]):
            mat = poses[index]
            if (
                pcd_name is None
                or pcd_name not in self.pcd_dict
                or len(self.pcd_dict[pcd_name]) <= index
            ):
                mesh = copy.deepcopy(meshes[index])
                mesh.scale(scales[index], center=np.array([0, 0, 0]))
                sampled_cloud = mesh.sample_points_poisson_disk(
                    number_of_points=num_pts[index]
                )
                cloud_points = np.asarray(sampled_cloud.points)
                if pcd_name not in self.pcd_dict:
                    self.pcd_dict[pcd_name] = []
                self.pcd_dict[pcd_name].append(cloud_points)
            else:
                cloud_points = self.pcd_dict[pcd_name][index]

            tf_obj_to_link = offsets[index]
            mat = mat @ tf_obj_to_link
            all_pc.append(cloud_points @ mat[:3, :3].T + mat[:3, 3])
        return np.concatenate(all_pc, axis=0)

    def compute_robot_pcd(
        self,
        qpos: np.ndarray,
        link_names: Optional[list[str]] = None,
        num_pts: Optional[list[int]] = None,
        pcd_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compute point cloud of robot links given joint positions."""
        self._load_robot_meshes()
        self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = list(self.meshes.keys())
        if num_pts is None:
            num_pts = [500] * len(link_names)

        link_idx_ls = [self.link_name_to_idx[link_name] for link_name in link_names]
        link_pose_ls = np.stack(
            [
                self.robot_model.get_link_pose(link_idx).to_transformation_matrix()
                for link_idx in link_idx_ls
            ]
        )
        return self._mesh_poses_to_pc(
            poses=link_pose_ls,
            meshes=[self.meshes[link_name] for link_name in link_names],
            offsets=[self.offsets[link_name] for link_name in link_names],
            num_pts=num_pts,
            scales=[self.scales[link_name] for link_name in link_names],
            pcd_name=pcd_name,
        )

    def compute_robot_meshes(
        self,
        qpos: np.ndarray,
        link_names: Optional[list[str]] = None,
    ) -> list:
        """Compute meshes of robot links given joint positions."""
        self._load_robot_meshes()
        self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = list(self.meshes.keys())

        link_idx_ls = [self.link_name_to_idx[link_name] for link_name in link_names]
        link_pose_ls = np.stack(
            [
                self.robot_model.get_link_pose(link_idx).to_transformation_matrix()
                for link_idx in link_idx_ls
            ]
        )
        meshes_ls = []
        for link_idx, link_name in enumerate(link_names):
            mesh = copy.deepcopy(self.meshes[link_name])
            mesh.scale(0.001, center=np.array([0, 0, 0]))
            mesh.transform(link_pose_ls[link_idx] @ self.offsets[link_name])
            meshes_ls.append(mesh)
        return meshes_ls

    def compute_fk_from_link_idx(
        self,
        qpos: np.ndarray,
        link_idx: list[int],
    ) -> list[np.ndarray]:
        """Compute forward kinematics of robot links given joint positions."""
        self.robot_model.compute_forward_kinematics(qpos)
        return [
            self.robot_model.get_link_pose(index).to_transformation_matrix()
            for index in link_idx
        ]

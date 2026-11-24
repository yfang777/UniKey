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

    def compute_fk_from_link_names(
        self,
        qpos: np.ndarray,
        link_names: list[str],
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics of robot links given joint positions."""
        self.robot_model.compute_forward_kinematics(qpos)
        return {
            name: self.robot_model.get_link_pose(self.link_name_to_idx[name]).to_transformation_matrix()
            for name in link_names
        }

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


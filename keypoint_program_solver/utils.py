"""Shared scene, camera, keypoint, and trajectory utilities."""

import datetime
import xml.etree.ElementTree as ET
from functools import partial, update_wrapper
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R, Slerp

from . import transform_utils as T



def resolve_robot_paths(robot_variant: str) -> tuple[Path, Path]:
    """Return the matching kinematics URDF and simulation XML."""
    if robot_variant != "w_wrist":
        raise ValueError(f"Unknown robot_variant: {robot_variant!r}")
    asset_dir = Path(__file__).resolve().parents[1] / "assets/xarm7"
    return (
        asset_dir / f"xarm7_{robot_variant}.urdf",
        asset_dir / f"xarm7_{robot_variant}.xml",
    )


def scene_model_text(model, name: str) -> str:
    """Read required text metadata embedded in a saved MuJoCo scene."""
    text_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if text_id == -1:
        raise ValueError(f"Scene has no {name}; regenerate it using the task configuration.")
    start = model.text_adr[text_id]
    size = model.text_size[text_id]
    return model.text_data[start : start + size - 1].decode("utf-8")


def camera_element(name: str, params: dict) -> ET.Element:
    """Convert OpenCV camera-to-parent calibration into a MuJoCo camera."""
    transform = np.asarray(params["extrinsic"], dtype=float)
    intrinsic = np.asarray(params["intrinsic"], dtype=float)
    width, height = params["resolution"]
    if transform.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError(f"{name}: expected a 4x4 extrinsic and 3x3 intrinsic")
    if width <= 0 or height <= 0 or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError(f"{name}: resolution and focal lengths must be positive")
    if intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0:
        raise ValueError(f"{name}: MuJoCo cameras do not support intrinsic skew")
    rotation = transform[:3, :3] @ np.diag([1, -1, -1])
    quat = R.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    return ET.Element("camera", {
        "name": name,
        "mode": "fixed",
        "pos": " ".join(map(str, transform[:3, 3])),
        "quat": " ".join(map(str, quat)),
        "resolution": f"{width} {height}",
        "sensorsize": "1 1",
        "focalpixel": f"{intrinsic[0, 0]} {intrinsic[1, 1]}",
        "principalpixel": f"{(width - 1) / 2 - intrinsic[0, 2]} {intrinsic[1, 2] - (height - 1) / 2}",
    })


def camera_intrinsic(model, camera_id: int, width: int, height: int) -> np.ndarray:
    """Return pixel intrinsics for a render at the requested dimensions."""
    fx, fy, px, py = model.cam_intrinsic[camera_id] / np.tile(model.cam_sensorsize[camera_id], 2)
    return np.array([
        [fx * width, 0, (width - 1) / 2 - px * width],
        [0, fy * height, (height - 1) / 2 + py * height],
        [0, 0, 1],
    ])


def concatenate_camera_frames(frames):
    """Join native camera images for a video, padding shorter images below."""
    height = max(frame.shape[0] for frame in frames)
    return np.concatenate([np.pad(frame, ((0, height - frame.shape[0]), (0, 0), (0, 0)))
                           for frame in frames], axis=1)


class CameraRenderers:
    """Own one MuJoCo renderer per image size, shared by cameras of that size."""
    def __init__(self, model):
        self.model = model
        self.renderers = {}

    def for_camera(self, camera_name):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Unknown scene camera: {camera_name}")
        width, height = map(int, self.model.cam_resolution[camera_id])
        size = (width, height)
        if size not in self.renderers:
            self.renderers[size] = mujoco.Renderer(self.model, height, width)
        return self.renderers[size]

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()


def bind_episode_constraints(module, episode_context):
    if not getattr(module, "uses_episode_context", False):
        return module
    context = np.array(episode_context, dtype=float, copy=True)
    if context.shape != (33,) or not np.isfinite(context).all():
        raise ValueError("Expected the 33-value shoe placement context")
    context.setflags(write=False)
    members = dict(vars(module))
    for name, function in vars(module).items():
        if name.startswith("stage") and "_constraint" in name and callable(function):
            members[name] = update_wrapper(partial(function, context=context), function)
    members["uses_episode_context"] = False
    return SimpleNamespace(**members)


def _resolve_xml_reference(base_dir, file_path):
    path = Path(file_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

def _append_body_site(body_to_sites, body_name, site_name):
    if site_name and body_name:
        body_to_sites.setdefault(body_name, [])
        if site_name not in body_to_sites[body_name]:
            body_to_sites[body_name].append(site_name)

def _collect_body_sites(root):
    body_to_sites = {}

    def traverse_bodies(body_elem, inherited_name=None):
        body_name = body_elem.attrib.get("name", inherited_name)
        for site in body_elem.findall("site"):
            _append_body_site(body_to_sites, body_name, site.attrib.get("name"))
        for child_body in body_elem.findall("body"):
            traverse_bodies(child_body, body_name)

    for top_body in root.findall("./worldbody/body"):
        traverse_bodies(top_body)

    return body_to_sites

def get_body_site_map(xml_path):
    xml_path = Path(xml_path)
    base_dir = xml_path.parent
    root = ET.parse(xml_path).getroot()

    body_to_sites = {}
    for include in root.findall(".//include"):
        include_file = include.attrib["file"]
        full_path = _resolve_xml_reference(base_dir, include_file)
        if not full_path.exists():
            print(f"⚠️ Warning: Include file {full_path} does not exist.")
            continue

        include_sites = _collect_body_sites(ET.parse(full_path).getroot())
        for body_name, site_names in include_sites.items():
            for site_name in site_names:
                _append_body_site(body_to_sites, body_name, site_name)

    if body_to_sites:
        return body_to_sites

    return _collect_body_sites(root)

def _mesh_path_and_scale(mesh_elem, base_dir):
    mesh_name = mesh_elem.attrib["name"]
    file_path = mesh_elem.attrib["file"]
    if "_replicate" in mesh_name:
        file_path = file_path.replace(mesh_name, mesh_name.split("_replicate")[0])
    mesh_path = _resolve_xml_reference(base_dir, file_path)
    scale = [float(value) for value in mesh_elem.attrib.get("scale", "1 1 1").split()]
    return mesh_path, scale

def _first_object_mesh_asset(root, base_dir):
    for mesh_elem in root.findall(".//asset/mesh"):
        mesh_name = mesh_elem.attrib.get("name")
        if mesh_name and "part" not in mesh_name:
            return _mesh_path_and_scale(mesh_elem, base_dir)
    return None

def _body_object_record(body_elem, mesh_path, scale):
    body_name = body_elem.attrib["name"]
    return {
        "name": body_name,
        "mesh_file": str(mesh_path),
        "initial_pos": [float(v) for v in body_elem.attrib.get("pos", "0 0 0").split()],
        "initial_quat": [float(v) for v in body_elem.attrib.get("quat", "1 0 0 0").split()],
        "scale": scale,
    }

def _object_map_from_included_root(root, base_dir):
    object_mesh = _first_object_mesh_asset(root, base_dir)
    if object_mesh is None:
        return {}

    mesh_path, scale = object_mesh
    object_map = {}
    for body_elem in root.findall("./worldbody/body"):
        if "name" in body_elem.attrib:
            body_name = body_elem.attrib["name"]
            object_map[body_name] = _body_object_record(body_elem, mesh_path, scale)
            break
    return object_map

def _asset_meshes(root, base_dir):
    meshes = {}
    for mesh_elem in root.findall(".//asset/mesh"):
        mesh_name = mesh_elem.attrib.get("name")
        if mesh_name:
            meshes[mesh_name] = _mesh_path_and_scale(mesh_elem, base_dir)
    return meshes

def _direct_body_mesh_name(body_elem):
    for geom_elem in body_elem.findall("geom"):
        mesh_name = geom_elem.attrib.get("mesh")
        if mesh_name and "part" not in mesh_name:
            return mesh_name
    return None

def _object_map_from_flattened_root(root, base_dir):
    meshes = _asset_meshes(root, base_dir)
    object_map = {}
    for body_elem in root.findall("./worldbody/body"):
        body_name = body_elem.attrib.get("name")
        if body_name in ("ground", None) or body_elem.attrib.get("childclass") == "xarm7":
            continue

        mesh_name = _direct_body_mesh_name(body_elem)
        if mesh_name is None:
            visual_mesh_name = f"{body_name}_mesh"
            if visual_mesh_name not in meshes:
                continue
            mesh_name = visual_mesh_name

        mesh_path, scale = meshes[mesh_name]
        object_map[body_name] = _body_object_record(body_elem, mesh_path, scale)
    return object_map

def get_object_map(scene_xml_path):
    scene_xml_path = Path(scene_xml_path)
    base_dir = scene_xml_path.parent
    scene_root = ET.parse(scene_xml_path).getroot()

    object_map = {}
    for include in scene_root.findall(".//include"):
        include_file = include.attrib.get("file")
        if not include_file:
            continue

        object_xml_path = _resolve_xml_reference(base_dir, include_file)
        if not object_xml_path.exists():
            print(f"⚠️ Warning: Include file {object_xml_path} does not exist.")
            continue

        include_root = ET.parse(object_xml_path).getroot()
        object_map.update(_object_map_from_included_root(include_root, object_xml_path.parent))

    if object_map:
        return object_map

    return _object_map_from_flattened_root(scene_root, base_dir)

def angle_between_quats(q1, q2):
    """Angle between two quaternions"""
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1, 1))

@njit(cache=True, fastmath=True)
def quat_slerp_jitted(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.
    (adapted from deoxys)
    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    EPS = 1e-8
    q0 = quat0 / np.linalg.norm(quat0)
    q1 = quat1 / np.linalg.norm(quat1)
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if np.abs(np.abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    if d < -1.0:
        d = -1.0
    elif d > 1.0:
        d = 1.0
    angle = np.arccos(d)
    if np.abs(angle) < EPS:
        return q0
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - fraction) * angle) * isin
    q1 *= np.sin(fraction * angle) * isin
    q0 += q1
    return q0

def _to_mat(control_points):
    if isinstance(control_points, list):
        control_points = np.array(control_points)

    n_points = control_points.shape[0]

    if control_points.shape[1] == 6:
        control_points_pos = control_points[:, :3]  # [N, 3]
        control_points_euler = control_points[:, 3:]  # [N, 3]
        control_points_rotmat = T.euler2mat(control_points_euler)  # [N, 3, 3]
        original_representation = "euler"

    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        control_points_pos = control_points[:, :3, 3]  # [N, 3]
        control_points_rotmat = control_points[:, :3, :3]  # [N, 3, 3]
        original_representation = "mat"

    elif control_points.shape[1] == 7:
        control_points_pos = control_points[:, :3]
        control_point_quat = control_points[:, 3:]
        control_points_rotmat = T.quat2mat(control_point_quat)

        original_representation = "quat_xyzw"
    else:
        raise ValueError('control_points not recognized')

    control_points_mat = np.eye(4)[np.newaxis, ...].repeat(n_points, axis=0)
    control_points_mat[:, :3, :3] = control_points_rotmat
    control_points_mat[:, :3, 3] = control_points_pos
    return control_points_mat, original_representation

def _mat_to_original_representation(control_points, original_representation):
    if isinstance(control_points, list):
        control_points = np.array(control_points)

    assert control_points.ndim == 3 and control_points.shape[1:] == (4, 4), "Input must be [N, 4, 4]"

    n_points = control_points.shape[0]
    pos = control_points[:, :3, 3]
    rot_mat = control_points[:, :3, :3]

    if original_representation == "quat_xyzw":
        quat = T.mat2quat(rot_mat)
        return np.concatenate([pos, quat], axis=1)

    elif original_representation == "euler":
        euler = T.mat2euler(rot_mat)
        return np.concatenate([pos, euler], axis=1)

    elif original_representation == "mat":
        return control_points

    else:
        raise ValueError(f"Representation '{original_representation}' not recognized")

def get_samples_jitted(control_points, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size):

    control_points_mat, original_represenation = _to_mat(control_points)
    n_points = control_points_mat.shape[0]

    num_samples_per_segment = np.empty(len(control_points_mat) - 1, dtype=np.int64)
    for i in range(len(control_points_mat) - 1):
        start_pos = control_points_mat[i, :3, 3]
        start_rotmat = control_points_mat[i, :3, :3]
        end_pos = control_points_mat[i+1, :3, 3]
        end_rotmat = control_points_mat[i+1, :3, :3]
        pos_diff = np.linalg.norm(start_pos - end_pos)
        rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
        pos_num_steps = np.ceil(pos_diff / opt_interpolate_pos_step_size)
        rot_num_steps = np.ceil(rot_diff / opt_interpolate_rot_step_size)
        num_path_poses = int(max(pos_num_steps, rot_num_steps))
        num_path_poses = max(num_path_poses, 2)  # at least 2 poses, start and end
        num_samples_per_segment[i] = num_path_poses

    # fill in samples
    control_points_quat = np.zeros((n_points, 7))
    control_points_quat[:, :3] = control_points_mat[:, :3, 3]
    control_points_quat[:, 3:] = T.mat2quat(control_points_mat[:, :3, :3])

    num_samples = num_samples_per_segment.sum()
    samples_7 = np.empty((num_samples, 7))
    sample_idx = 0
    for i in range(len(control_points_quat) - 1):
        start_pos, start_xyzw = control_points_quat[i, :3], control_points_quat[i, 3:]
        end_pos, end_xyzw = control_points_quat[i+1, :3], control_points_quat[i+1, 3:]
        # using proper quaternion slerp interpolation
        poses_7 = np.empty((num_samples_per_segment[i], 7))
        for j in range(num_samples_per_segment[i]):
            alpha = j / (num_samples_per_segment[i] - 1)
            pos = start_pos * (1 - alpha) + end_pos * alpha
            blended_xyzw = quat_slerp_jitted(start_xyzw, end_xyzw, alpha)
            pose_7 = np.empty(7)
            pose_7[:3] = pos
            pose_7[3:] = blended_xyzw
            poses_7[j] = pose_7
        samples_7[sample_idx:sample_idx+num_samples_per_segment[i]] = poses_7
        sample_idx += num_samples_per_segment[i]
    assert num_samples >= 2, f'num_samples: {num_samples}'

    sample_points_mat, _ = _to_mat(samples_7)
    sample_points_original_representation = _mat_to_original_representation(sample_points_mat, original_represenation)
    return sample_points_original_representation, num_samples

@njit(cache=True, fastmath=True)
def angle_between_rotmat(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    return np.arccos(cos_theta)

def get_linear_interpolation_steps(start_pose, end_pose, pos_step_size, rot_step_size):
    """
    Given start and end pose, calculate the number of steps to interpolate between them.
    Args:
        start_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        end_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        pos_step_size: position step size
        rot_step_size: rotation step size
    Returns:
        num_path_poses: number of poses to interpolate
    """
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    pos_diff = np.linalg.norm(start_pos - end_pos)
    rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
    pos_num_steps = np.ceil(pos_diff / pos_step_size)
    rot_num_steps = np.ceil(rot_diff / rot_step_size)
    num_path_poses = int(max(pos_num_steps, rot_num_steps))
    num_path_poses = max(num_path_poses, 2)  # at least start and end poses
    return num_path_poses

def linear_interpolate_poses(start_pose, end_pose, num_poses):
    """
    Interpolate between start and end pose.
    """
    assert num_poses >= 2, 'num_poses must be at least 2'
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]

        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    slerp = Slerp([0, 1], R.from_matrix([start_rotmat, end_rotmat]))
    poses = []
    for i in range(num_poses):
        alpha = i / (num_poses - 1)
        pos = start_pos * (1 - alpha) + end_pos * alpha
        rotmat = slerp(alpha).as_matrix()
        if start_pose.shape == (6,):
            euler = T.mat2euler(rotmat)
            poses.append(np.concatenate([pos, euler]))
        elif start_pose.shape == (4, 4):
            pose = np.eye(4)
            pose[:3, :3] = rotmat
            pose[:3, 3] = pos
            poses.append(pose)
        elif start_pose.shape == (7,):
            quat = T.mat2quat(rotmat)
            pose = np.concatenate([pos, quat])
            poses.append(pose)
    return np.array(poses)

def spline_interpolate_poses(control_points, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size):
    """Interpolate the two end-effector targets used by the recorded controller."""
    matrices, representation = _to_mat(control_points)
    samples, _ = get_samples_jitted(matrices, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size)
    return _mat_to_original_representation(samples, representation)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'

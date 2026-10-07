from robot.cameras import camera_intrinsic
import os
import numpy as np
from numba import njit
import datetime
import scipy.interpolate as interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
from . import transform_utils as T
import yaml

# ---------- mujoco -------------
import os
import xml.etree.ElementTree as ET
from pathlib import Path
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




#################################


def normalize_vars(vars, og_bounds):
    """
    Given 1D variables and bounds, normalize the variables to [-1, 1] range.
    """
    normalized_vars = np.empty_like(vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        normalized_vars[i] = (vars[i] - b_min) / (b_max - b_min) * 2 - 1
    return normalized_vars

def unnormalize_vars(normalized_vars, og_bounds):
    """
    Given 1D variables in [-1, 1] and original bounds, denormalize the variables to the original range.
    """
    vars = np.empty_like(normalized_vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        vars[i] = (normalized_vars[i] + 1) / 2 * (b_max - b_min) + b_min
    return vars

def farthest_point_sampling(pc, num_points):
    """
    Given a point cloud, sample num_points points that are the farthest apart.
    Use o3d farthest point sampling.
    """
    import open3d as o3d

    assert pc.ndim == 2 and pc.shape[1] == 3, "pc must be a (N, 3) numpy array"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    downpcd_farthest = pcd.farthest_point_down_sample(num_points)
    return np.asarray(downpcd_farthest.points)

def mesh_prim_mesh_to_trimesh_mesh(mesh_prim):
    """
    Convert a prim mesh to a trimesh mesh.
    """
    trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh_prim)
    return trimesh_object


@njit(cache=True, fastmath=True)
def batch_transform_points(points, transforms):
    """
    Apply multiple of transformation to point cloud, return results of individual transformations.
    Args:
        points: point cloud (N, 3).
        transforms: M 4x4 transformations (M, 4, 4).
    Returns:
        np.array: point clouds (M, N, 3).
    """
    assert transforms.shape[1:] == (4, 4), 'transforms must be of shape (M, 4, 4)'
    transformed_points = np.zeros((transforms.shape[0], points.shape[0], 3))
    for i in range(transforms.shape[0]):
        pos, R = transforms[i, :3, 3], transforms[i, :3, :3]
        transformed_points[i] = np.dot(points, R.T) + pos
    return transformed_points

def calculate_collision_cost(poses, sdf_func, collision_points, threshold):
    '''
    sdf_func:
        A function that takes [M, 3] coordinates and returns their signed distance to the nearest obstacle.
        Positive: outside
        Zero: on the surface
        Negative: inside the obstacle
    collision_points: np.array (centered)
        A (N, 3) array of collision points.
    threshold: float (0.2)
        A threshold to add to the signed distance to avoid numerical issues.
    '''
    assert poses.shape[1:] == (4, 4)
    transformed_pcs = batch_transform_points(collision_points, poses)
    transformed_pcs_flatten = transformed_pcs.reshape(-1, 3)  # [num_poses * num_points, 3]
    signed_distance = sdf_func(transformed_pcs_flatten) + threshold  # [num_poses * num_points]
    signed_distance = signed_distance.reshape(-1, collision_points.shape[0])  # [num_poses, num_points]
    non_zero_mask = signed_distance > 0
    collision_cost = np.sum(signed_distance[non_zero_mask])
    return collision_cost

def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Filter points by taking only points within workspace bounds.
    """
    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask

def transform_keypoints(transform, keypoints, movable_mask):
    assert transform.shape == (4, 4)
    transformed_keypoints = keypoints.copy()
    if movable_mask.sum() > 0:
        transformed_keypoints[movable_mask] = np.dot(keypoints[movable_mask], transform[:3, :3].T) + transform[:3, 3]
    return transformed_keypoints


import mujoco
import einops as eo
def _calc_cammatrices(data, camera_name, camera_dict):
    camera_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    weird_R = eo.rearrange(data.cam_xmat[camera_id], '(i j) -> i j', i=3, j=3).T
    R = np.eye(3)
    R[0, :] = weird_R[0, :]
    R[1, :] = - weird_R[1, :]
    R[2, :] = - weird_R[2, :]
    cam_pos = data.cam_xpos[camera_id]
    t = -np.dot(R, cam_pos)

    ex_mat = np.eye(4)
    ex_mat[:3, :3] = R
    ex_mat[:3, 3] = t

    width, height = data.model.cam_resolution[camera_id]
    in_mat = np.vstack([camera_intrinsic(data.model, camera_id, width, height), np.zeros(3)])

    return ex_mat, in_mat


def visualize_scene_and_sdf_plotly(scene_points, sdf_voxels, resolution, bounds_min, bounds_max):
    """
    Visualize both scene points and SDF voxels using Plotly.

    Args:
        scene_points: np.ndarray of shape (N, 3) containing scene points
        sdf_voxels: np.ndarray of shape (X, Y, Z) containing SDF values
        resolution: float, the resolution used for SDF computation
        bounds_min: np.ndarray of shape (3,) containing minimum bounds
        bounds_max: np.ndarray of shape (3,) containing maximum bounds
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    # Create subplot figure
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'volume'}]])

    # Convert normalized scene points back to world coordinates
    scene_points_centered = (scene_points - bounds_min) / (bounds_max - bounds_min)

    # Add scene points


    fig.add_trace(
        go.Scatter3d(
            x=scene_points[:, 0],
            y=scene_points[:, 1],
            z=scene_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity=0.5
            ),
            name='Scene Points'
        ),
        row=1, col=1
    )


    # Create grid for SDF visualization
    x, y, z = np.meshgrid(
        np.arange(sdf_voxels.shape[0]),
        np.arange(sdf_voxels.shape[1]),
        np.arange(sdf_voxels.shape[2]),
        indexing='ij'
    )

    # Convert grid coordinates to world coordinates
    x_world = x * resolution + bounds_min[0]
    y_world = y * resolution + bounds_min[1]
    z_world = z * resolution + bounds_min[2]

    # Add SDF volume
    fig.add_trace(
        go.Volume(
            x=x_world.flatten(),
            y=y_world.flatten(),
            z=z_world.flatten(),
            value=sdf_voxels.flatten(),
            isomin=-0.1,
            isomax=0.1,
            opacity=0.3,
            surface_count=20,
            colorscale='blues',
            colorbar=dict(title='SDF Value'),
            name='SDF Volume'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title='Scene Points and SDF Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1500,
        height=800
    )

    fig.write_html("scene_and_sdf_visualization.html")
    print("✅ Saved visualization to scene_and_sdf_visualization.html")



#  ===================== Transform utils =====================
def angle_between_quats(q1, q2):
    """Angle between two quaternions"""
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1, 1))

@njit(cache=True, fastmath=True)
def angle_between_rotmat(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    return np.arccos(cos_theta)

@njit(cache=True, fastmath=True)
def path_length(samples_homo):
    assert samples_homo.shape[1:] == (4, 4), 'samples_homo must be of shape (N, 4, 4)'
    pos_length = 0
    rot_length = 0
    for i in range(len(samples_homo) - 1):
        pos_length += np.linalg.norm(samples_homo[i, :3, 3] - samples_homo[i+1, :3, 3])
        rot_length += angle_between_rotmat(samples_homo[i, :3, :3], samples_homo[i+1, :3, :3])
    return pos_length, rot_length

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


# @njit(cache=True, fastmath=True)
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

@njit(cache=True, fastmath=True)
def consistency(poses_a, poses_b, rot_weight=0.5):
    assert poses_a.shape[1:] == (4, 4) and poses_b.shape[1:] == (4, 4), 'poses must be of shape (N, 4, 4)'
    min_distances = np.zeros(len(poses_a), dtype=np.float64)
    for i in range(len(poses_a)):
        min_distance = 9999999
        a = poses_a[i]
        for j in range(len(poses_b)):
            b = poses_b[j]
            pos_distance = np.linalg.norm(a[:3, 3] - b[:3, 3])
            rot_distance = angle_between_rotmat(a[:3, :3], b[:3, :3])
            distance = pos_distance + rot_distance * rot_weight
            min_distance = min(min_distance, distance)
        min_distances[i] = min_distance
    return np.mean(min_distances)

def fit_b_spline(control_points):
    # determine appropriate k
    k = min(3, control_points.shape[0]-1)
    spline = interpolate.splprep(control_points.T, s=0, k=k)
    return spline

def sample_from_spline(spline, num_samples):
    sample_points = np.linspace(0, 1, num_samples)
    if isinstance(spline, RotationSpline):
        samples = spline(sample_points).as_matrix()  # [num_samples, 3, 3]
    else:
        assert isinstance(spline, tuple) and len(spline) == 2, 'spline must be a tuple of (tck, u)'
        tck, u = spline
        samples = interpolate.splev(np.linspace(0, 1, num_samples), tck)  # [spline_dim, num_samples]
        samples = np.array(samples).T  # [num_samples, spline_dim]
    return samples


def spline_interpolate_poses(control_points, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size):
    """
    Interpolate between through the control points using spline interpolation.
    1. Fit a b-spline through the positional terms of the control points.
    2. Fit a RotationSpline through the rotational terms of the control points.
    3. Sample the b-spline and RotationSpline at num_steps.

    Args:
        control_points: [N, 6] position + euler or [N, 4, 4] pose or [N, 7] position + quat
        num_steps: number of poses to interpolate
    Returns:
        poses: [num_steps, 6] position + euler or [num_steps, 4, 4] pose or [num_steps, 7] position + quat
    """

    control_points_mat, original_representation = _to_mat(control_points)

    # remove the duplicate points (threshold 1e-3)
    diff = np.linalg.norm(np.diff(control_points_mat[:, :3, 3], axis=0), axis=1)
    mask = diff > 1e-3
    mask = np.concatenate([[True], mask[:-1], [True]]) # always keep the first and last points

    control_points_mat_filter = control_points_mat[mask]

    # if there are not enough points, just linear interpolate (or repeat)
    if len(control_points_mat_filter) == 2 :
        # Linear Interpolation
        control_points_mat_filter_sample, num_samples = get_samples_jitted(control_points_mat_filter, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size)
    else:
        # B-spline Interplolation
        pos_spline = fit_b_spline(control_points_pos)
        num_steps = get_linear_interpolation_steps(
            control_points_pos[0], control_points_pos[-1],
            opt_interpolate_pos_step_size,
            opt_interpolate_rot_step_size)
        # fit RotationSpline through rotational terms control points
        times = pos_spline[1]
        rotations = R.from_matrix(control_points_rotmat)
        rot_spline = RotationSpline(times, rotations)
        # sample from the splines
        pos_samples = sample_from_spline(pos_spline, num_steps)  # [num_steps, 3]
        rot_samples = sample_from_spline(rot_spline, num_steps)  # [num_steps, 3, 3]

        control_points_mat_filter = np.eye(4)[np.newaxis, ...].repeat(len(control_points_mat_filter), axis=0)
        control_points_mat_filter[:, :3, :3] = rot_samples
        control_points_mat_filter[:, :3, 3] = pos_samples

    poses = _mat_to_original_representation(control_points_mat_filter, original_representation)
    return poses

# def spline_interpolate_poses(control_points, num_steps):
#     """
#     Interpolate between through the control points using spline interpolation.
#     1. Fit a b-spline through the positional terms of the control points.
#     2. Fit a RotationSpline through the rotational terms of the control points.
#     3. Sample the b-spline and RotationSpline at num_steps.

#     Args:
#         control_points: [N, 6] position + euler or [N, 4, 4] pose or [N, 7] position + quat
#         num_steps: number of poses to interpolate
#     Returns:
#         poses: [num_steps, 6] position + euler or [num_steps, 4, 4] pose or [num_steps, 7] position + quat
#     """
#     assert num_steps >= 2, 'num_steps must be at least 2'
#     if isinstance(control_points, list):
#         control_points = np.array(control_points)
#     if control_points.shape[1] == 6:
#         control_points_pos = control_points[:, :3]  # [N, 3]
#         control_points_euler = control_points[:, 3:]  # [N, 3]
#         control_points_rotmat = []
#         for control_point_euler in control_points_euler:
#             control_points_rotmat.append(T.euler2mat(control_point_euler))
#         control_points_rotmat = np.array(control_points_rotmat)  # [N, 3, 3]
#     elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
#         control_points_pos = control_points[:, :3, 3]  # [N, 3]
#         control_points_rotmat = control_points[:, :3, :3]  # [N, 3, 3]
#     elif control_points.shape[1] == 7:
#         control_points_pos = control_points[:, :3]
#         control_points_rotmat = []
#         for control_point_quat in control_points[:, 3:]:
#             control_points_rotmat.append(T.quat2mat(control_point_quat))
#         control_points_rotmat = np.array(control_points_rotmat)
#     else:
#         raise ValueError('control_points not recognized')
#     # remove the duplicate points (threshold 1e-3)
#     diff = np.linalg.norm(np.diff(control_points_pos, axis=0), axis=1)

#     # always keep the first and last points
#     # mask = diff > 1e-3
#     # mask = np.concatenate([[True], mask[:-1], [True]])

#     # always keep the last point, and any point that is far enough from the next point
#     mask = np.concatenate([diff > 1e-4, [True]])

#     control_points_pos = control_points_pos[mask]
#     control_points_rotmat = control_points_rotmat[mask]

#     # if there are not enough points, just linear interpolate (or repeat)
#     if len(control_points_pos) < 2:

#         return get_samples_jitted
#         # duplicate the single point to make it at least 2
#         rotations = R.from_matrix(control_points_rotmat)
#         rot_spline = RotationSpline(2, rotations)
#         pos_samples = np.concatenate([control_points_pos, control_points_pos], axis=0)
#         rot_samples = sample_from_spline(rot_spline, 2)
#     else:
#         # fit b-spline through positional terms control points
#         pos_spline = fit_b_spline(control_points_pos)
#         # fit RotationSpline through rotational terms control points
#         times = pos_spline[1]
#         rotations = R.from_matrix(control_points_rotmat)
#         rot_spline = RotationSpline(times, rotations)
#         # sample from the splines
#         pos_samples = sample_from_spline(pos_spline, num_steps)  # [num_steps, 3]
#         rot_samples = sample_from_spline(rot_spline, num_steps)  # [num_steps, 3, 3]

#     pos_spline = fit_b_spline(control_points_pos)
#     # fit RotationSpline through rotational terms control points
#     times = pos_spline[1]
#     rotations = R.from_matrix(control_points_rotmat)
#     rot_spline = RotationSpline(times, rotations)
#     # sample from the splines
#     pos_samples = sample_from_spline(pos_spline, num_steps)  # [num_steps, 3]
#     rot_samples = sample_from_spline(rot_spline, num_steps)  # [num_steps, 3, 3]
#     if control_points.shape[1] == 6:
#         poses = []
#         for i in range(num_steps):
#             pose = np.concatenate([pos_samples[i], T.mat2euler(rot_samples[i])])
#             poses.append(pose)
#         poses = np.array(poses)
#     elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
#         poses = np.empty((num_steps, 4, 4))
#         poses[:, :3, :3] = rot_samples
#         poses[:, :3, 3] = pos_samples
#         poses[:, 3, 3] = 1
#     elif control_points.shape[1] == 7:
#         poses = np.empty((num_steps, 7))
#         for i in range(num_steps):
#             quat = T.mat2quat(rot_samples[i])
#             pose = np.concatenate([pos_samples[i], quat])
#             poses[i] = pose
#     return poses

# ===================== Debug utils =====================

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





def get_callable_grasping_cost_fn(env):
    def get_grasping_cost(keypoint_idx):
        return -env.is_grasping_keypoint(keypoint_idx=keypoint_idx) + 1
        # keypoint_object = env.get_object_by_keypoint(keypoint_idx)
        # return -env.is_grasping(candidate_obj=keypoint_object) + 1  # return 0 if grasping an object, 1 if not grasping any object
    return get_grasping_cost

def merge_dicts(dicts):
    return {
        k : v
        for d in dicts
        for k, v in d.items()
    }


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e

def load_functions_from_txt(txt_path, get_grasping_cost_fn):
    if txt_path is None:
        return []
    # load txt file
    with open(txt_path, 'r') as f:
        functions_text = f.read()
    # execute functions
    gvars_dict = {
        'np': np,
        'get_grasping_cost_by_keypoint_idx': get_grasping_cost_fn,
    }  # external library APIs
    lvars_dict = dict()
    exec_safe(functions_text, gvars=gvars_dict, lvars=lvars_dict)
    return list(lvars_dict.values())

def print_opt_debug_dict(debug_dict, save_path=None):
    lines = []
    lines.append('\n' + '#' * 40)
    lines.append(f'# Optimization debug info:')
    max_key_length = max(len(str(k)) for k in debug_dict.keys())
    for k, v in debug_dict.items():
        if isinstance(v, (int, float)):
            lines.append(f'# {k:<{max_key_length}}: {v:.05f}')
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            lines.append(f'# {k:<{max_key_length}}: {np.array(v).round(5)}')
        else:
            lines.append(f'# {k:<{max_key_length}}: {v}')
    lines.append('#' * 40 + '\n')

    # Print all lines
    for line in lines:
        print(line)
    # Optionally, save to file
    if save_path is not None:
        with open(save_path, 'a') as f:
            for line in lines:
                f.write(str(line) + '\n')

def clean_txt(txt_path):
    """Empties the file at txt_path."""
    with open(txt_path, 'w') as f:
        pass

def log_str(string, txt_path):
    """Appends a string to a text file, followed by a newline."""
    with open(txt_path, 'a') as f:
        f.write(str(string) + '\n')
        print(string)

def log_array(arr, dir, name):
    """Save a numpy array to a .npy file in the specified directory with the given name."""
    if not os.path.exists(dir):
        os.makedirs(dir)
    file_path = os.path.join(dir, name)
    np.save(file_path, arr)


import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

def get_all_included_files(xml_path):
    """Recursively get all included files from a base MJCF scene XML"""
    xml_path = Path(xml_path)
    base_dir = xml_path.parent
    all_files = []

    def recurse(path):
        full_path = base_dir / path
        if not full_path.exists():
            print(f"⚠️ Skipping missing file: {full_path}")
            return
        all_files.append(full_path.resolve())
        tree = ET.parse(full_path)
        root = tree.getroot()
        for include in root.findall(".//include"):
            included_file = include.attrib["file"]
            recurse(included_file)

    recurse(xml_path.name)
    return all_files


def extract_keypoint_descriptions_from_file(scene_xml_path):
    """Find mesh .stl file → convert to .json → load keypoint descriptions"""
    mesh_jsons = []
    keypoints_dict = {}
    base_dir = scene_xml_path.parent

    tree = ET.parse(scene_xml_path)
    root = tree.getroot()
    keypoint_dicts = {}

    for mesh in root.findall(".//mesh"):
        mesh_file = mesh.attrib.get("file")
        if not mesh_file.endswith(".stl"):
            continue

        mesh_json = Path(mesh_file).with_suffix(".json")
        full_json_path = (base_dir / mesh_json).resolve()
        if not full_json_path.exists():
            print(f"⚠️ Missing keypoint file: {full_json_path}")
            continue

        with open(full_json_path, "r") as f:
            keypoints = json.load(f)["keypoints"]

        for kp in keypoints:
            idx = str(kp.get("source_keypoint", len(keypoint_dicts)))
            desc = kp.get("description", "unknown")
            keypoint_dicts[idx] = {
                "description": f"{desc}",
            }

    return keypoint_dicts


def get_keypoint_description_dict(scene_xml_path):
    all_mjcf_paths = get_all_included_files(scene_xml_path)

    global_keypoint_dict = {
        "0": {
            "description": "end effector",
        }
    }

    keypoint_idx = 1  # Start after end effector

    for mjcf_path in all_mjcf_paths:
        per_file_kp_dict = extract_keypoint_descriptions_from_file(mjcf_path)
        for _, entry in per_file_kp_dict.items():
            global_keypoint_dict[str(keypoint_idx)] = entry
            keypoint_idx += 1

    return global_keypoint_dict


def get_keypoint_prompt(scene_xml_path):
    keypoint_description_dict = get_keypoint_description_dict(scene_xml_path)
    keypoint_id = 0
    keypoint_prompt = ""
    for key, value in keypoint_description_dict.items():
        keypont_name = keypoint_description_dict[key]["description"]
        keypoint_prompt += f"Keypoint {keypoint_id}: {keypont_name} \n"
        keypoint_id += 1
    return keypoint_prompt

# --------------------------------------------------------

import re
from typing import Dict, List, Any, Tuple

_STAGE_HDR_RE = re.compile(
    r"^###\s*stage\s*(\d+)\s*constraints?\s*\(.*?\)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_GRASP_RE = re.compile(r"grasp_stage\s*=\s*(\[[^\]]*\])", flags=re.IGNORECASE | re.DOTALL)
_MACROS_HDR_RE = re.compile(r"^#\s*[-=]*\s*Macros\b.*?$", flags=re.IGNORECASE | re.MULTILINE)


def _first_code_block(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    return m.group(1) if m else text


def _parse_grasp_stage(text_or_code: str) -> List[int]:
    m = _GRASP_RE.search(text_or_code)
    if not m:
        raise ValueError("extract_output: missing 'grasp_stage = [...]'")
    arr_src = m.group(1)
    return [int(x) for x in re.findall(r"-?\d+", arr_src)]


def _strip_grasp_line(code: str) -> str:
    return _GRASP_RE.sub("", code, count=1)


def _extract_macros(code: str) -> Tuple[str, str]:
    m = _MACROS_HDR_RE.search(code)
    if not m:
        return "", code
    start = m.start()
    # Look for either stage headers or first constraint function
    stage_matches = list(_STAGE_HDR_RE.finditer(code))
    constraint_pat = re.compile(r"^\s*def\s+stage\d+_constraint", flags=re.MULTILINE)
    constraint_matches = list(constraint_pat.finditer(code))

    # Use whichever comes first - stage header or constraint function
    stage_start = stage_matches[0].start() if stage_matches else len(code)
    constraint_start = constraint_matches[0].start() if constraint_matches else len(code)
    end = min(stage_start, constraint_start)

    macros_src = code[start:end].strip()
    code_wo_macros = (code[:start] + code[end:]).strip()
    return macros_src, code_wo_macros


def _stage_chunks(code: str) -> List[Tuple[int, str]]:
    chunks = []
    matches = list(_STAGE_HDR_RE.finditer(code))
    if not matches:
        return chunks
    for i, m in enumerate(matches):
        stage_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
        chunk = code[start:end].strip()
        if chunk:
            chunks.append((stage_num, chunk))
    return chunks


def _collect_fn_names(stage_code: str, stage_num: int) -> List[str]:
    pat = re.compile(rf"^\s*def\s+(stage{stage_num}_constraint[\w\d_]*)\s*\(",
                     flags=re.MULTILINE)
    return pat.findall(stage_code)


def _collect_all_constraint_funcs(code: str) -> List[Tuple[str, int, int, int]]:
    """
    Return list of tuples: (func_name, stage_num, start_idx, end_idx_placeholder)
    end_idx will be filled later based on next match.
    """
    pat = re.compile(r"^\s*def\s+(stage(\d+)_constraint[\w\d_]*)\s*\(", flags=re.MULTILINE)
    matches = list(pat.finditer(code))
    funcs = []
    for m in matches:
        fn_name = m.group(1)
        stage_num = int(m.group(2))
        funcs.append((fn_name, stage_num, m.start(), m.end()))
    # compute source slice ends as next match.start() or len(code)
    out = []
    for i, (fn_name, stage_num, start, _) in enumerate(funcs):
        end = funcs[i + 1][2] if i + 1 < len(funcs) else len(code)
        out.append((fn_name, stage_num, start, end))
    return out


def extract_output(planner_text: str) -> Dict[str, Any]:
    """
    Parse a ChatGPT planner response and return:
      {
        "grasp_stage": [...],
        "num_stages": N,
        "script": "<full python module text>",
        "funcs": { <stage:int>: { <func_name>: <func_src>, ... }, ... }
      }

    Robust to:
      - Missing '### Stage X' headers (falls back to scanning functions)
      - Extra prose around code (pulls first fenced code block if present)
    """
    # 1) Pull code (first fenced block if present)
    code_raw = _first_code_block(planner_text).strip()

    # 2) Parse grasp_stage from the *entire* message (fallback to code)
    grasp_stage = _parse_grasp_stage(planner_text if _GRASP_RE.search(planner_text) else code_raw)
    num_stages = len(grasp_stage)

    # 3) Remove grasp_stage line from code to avoid duplicates when rebuilding script
    code_wo_grasp = _strip_grasp_line(code_raw).strip()

    # 4) (Optional) separate macros block (unused here, but preserves structure if needed later)
    # We don't drop macros from the final script; we keep original ordering.
    # The macros_src is not required by the requested return spec.
    _macros_src, _code_wo_macros = _extract_macros(code_wo_grasp)

    # 5) Collect all constraint function definitions with their source
    func_spans = _collect_all_constraint_funcs(code_wo_grasp)
    funcs_map: Dict[int, Dict[str, str]] = {}
    for fn_name, stage_num, start, end in func_spans:
        fn_src = code_wo_grasp[start:end].rstrip()
        funcs_map.setdefault(stage_num, {})[fn_name] = fn_src

    # 6) Build a clean, directly-importable script
    header = [
        "# Auto-generated planner script",
        "# This file was reconstructed by extract_output()",
        f"grasp_stage = {grasp_stage}",
        f"num_stages = {num_stages}",
        "",
    ]
    script = "\n".join(header) + code_wo_grasp + ("\n" if not code_wo_grasp.endswith("\n") else "")

    return {
        "grasp_stage": grasp_stage,
        "num_stages": num_stages,
        "script": script,
        "funcs": funcs_map,
    }

import random
import torch
import numpy as np
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

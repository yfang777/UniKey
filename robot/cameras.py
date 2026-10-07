"""Camera catalog conversion and native-resolution rendering."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


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
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
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

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from robot.cameras import camera_element
from .utils.default_world import write_scene_xml as write_base_scene_xml
from .utils.physics_config import finger_geom_attrs, normalize_physics_config, object_geom_attrs
from .utils.runtime_scene import RuntimeScene, runtime_scene_from_xml

REPO_ROOT = Path(__file__).resolve().parents[1]

def yaw_deg_to_quat_wxyz(yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=float)


def uniform_sample_params(bounds: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, bounds.shape[0]))


def format_vec(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def write_gripper_xml(
    sample_dir: Path,
    gripper_xml_path: Path,
    physics_cfg: dict,
) -> Path:
    root = ET.parse(gripper_xml_path).getroot()
    solver = normalize_physics_config(physics_cfg)["solver"]
    option = root.find("option")
    option.set("impratio", str(solver["impratio"]))
    option.set("noslip_iterations", str(solver["noslip_iterations"]))
    for asset in root.findall("./asset/*"):
        file_path = asset.attrib.get("file")
        if file_path is not None and not Path(file_path).is_absolute():
            asset.set("file", str((gripper_xml_path.parent / file_path).resolve()))

    attrs = finger_geom_attrs(physics_cfg)
    for default in root.findall(".//default"):
        if default.attrib.get("class") in {"pad_box1", "pad_box2"}:
            geom = default.find("geom")
            for attr_name, attr_value in attrs.items():
                geom.set(attr_name, attr_value)

    if "gripper" in physics_cfg:
        force_range = format_vec(physics_cfg["gripper"]["forcerange"])
        for name in ("gripper_right", "gripper_left"):
            actuator = root.find(f"./actuator/position[@name='{name}']")
            actuator.set("forcerange", force_range)
            actuator.set("forcelimited", "true")

    ET.indent(root, space="  ")
    xml_path = sample_dir / "gripper.xml"
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    return xml_path


class AbstractTaskSceneGenerator(ABC):
    def __init__(
        self,
        cfg: dict,
        gripper_xml_path: str | Path,
        object_dir: str | Path,
        physics_cfg: dict | None = None,
        *,
        cameras: dict,
    ) -> None:
        self.cfg = cfg
        self.cameras = cameras
        self.camera_names = tuple(cameras)
        self.gripper_xml_path = Path(gripper_xml_path).expanduser().resolve()
        robot_root = ET.parse(self.gripper_xml_path).getroot()
        self.robot_variant = robot_root.find("./custom/text[@name='robot_variant']").attrib["data"]
        self.object_dir = Path(object_dir).expanduser().resolve()
        self.physics_cfg = normalize_physics_config(physics_cfg)
        self.bounds = self.setup_bounds()

    @abstractmethod
    def setup_bounds(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def write_sample(self, sample_dir: Path, sample_params: np.ndarray) -> None:
        raise NotImplementedError

    def sample_params(self, n_samples: int, seed: int) -> np.ndarray:
        return uniform_sample_params(self.bounds, n_samples, seed)

    def generate_scene(self, sample_params: np.ndarray, output_cls_dir: str | Path, sample_idx: int) -> str:
        sample_dir = Path(output_cls_dir).expanduser().resolve() / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True)

        sample_params_array = np.asarray(sample_params, dtype=float)
        self.write_sample(sample_dir, sample_params_array)
        scene_path = sample_dir / "tmp_scene.xml"
        scene_root = ET.parse(scene_path).getroot()
        robot_path = write_gripper_xml(sample_dir, self.gripper_xml_path, self.physics_cfg)
        robot_root = ET.parse(robot_path).getroot()
        scene_root.find("include").set("file", str(robot_path))
        for name, params in self.cameras.items():
            parent_name = params["parent_frame"]
            parent = (scene_root.find("worldbody") if parent_name == "world"
                      else robot_root.find(f".//body[@name='{parent_name}']"))
            if parent is None:
                raise ValueError(f"Unknown camera parent_frame: {parent_name}")
            parent.append(camera_element(name, params))
        ET.ElementTree(robot_root).write(robot_path, encoding="unicode")
        visual = ET.SubElement(scene_root, "visual")
        ET.SubElement(visual, "global", offwidth=str(max(p["resolution"][0] for p in self.cameras.values())),
                      offheight=str(max(p["resolution"][1] for p in self.cameras.values())))
        if "lighting" in self.cfg:
            lighting = self.cfg["lighting"]
            ET.SubElement(visual, "headlight", active="0")
            worldbody = scene_root.find("./worldbody")
            for light in worldbody.findall("light"):
                worldbody.remove(light)
            for index, lamp in enumerate(lighting["lights"]):
                ET.SubElement(
                    worldbody, "light", name=f"room_light_{index + 1}",
                    pos=format_vec(lamp["pos"]), dir=format_vec(lamp["dir"]),
                    ambient=format_vec(lighting["ambient"]),
                    diffuse=format_vec(lighting["diffuse"]),
                    specular=format_vec(lighting["specular"]),
                    directional="false", castshadow="false", cutoff="90", exponent="0",
                )
        custom = ET.SubElement(scene_root, "custom")
        ET.SubElement(custom, "text", name="observation_cameras", data=" ".join(self.camera_names))
        ET.ElementTree(scene_root).write(scene_path, encoding="unicode")
        sample_metadata = {
            "robot_variant": self.robot_variant,
            "camera_names": list(self.camera_names),
            "camera_params": self.cameras,
            "sample_idx": int(sample_idx),
            "sample_params": sample_params_array.tolist(),
            "physics": self.physics_cfg,
        }
        (sample_dir / "sample_params.json").write_text(
            json.dumps(sample_metadata, indent=4),
            encoding="utf-8",
        )
        return str(sample_dir / "tmp_scene.xml")

    def generate_runtime_scene(
        self,
        sample_params: np.ndarray,
        output_cls_dir: str | Path,
        sample_idx: int,
    ) -> RuntimeScene:
        scene = runtime_scene_from_xml(self.generate_scene(sample_params, output_cls_dir, sample_idx))
        scene.metadata["physics"] = self.physics_cfg
        return scene

    def generate_samples(self, base_dir: str | Path, n_samples: int, seed: int) -> None:
        base_dir = Path(base_dir).expanduser().resolve()
        base_dir.mkdir(parents=True)

        for sample_id, sample_params in enumerate(self.sample_params(n_samples, seed)):
            self.generate_scene(sample_params, base_dir, sample_id)


def write_arrange_cube_by_color_scene(
    sample_dir: Path, cfg: dict, physics_cfg: dict, gripper_xml_path: Path,
    task_name: str, cube_color: str, plate_colors: tuple[str, ...],
    cube_pose: np.ndarray, plate_offsets: np.ndarray,
) -> None:
    asset_dir = resolve_repo_path(cfg["asset_dir"])
    info = json.loads((asset_dir / "asset_info.json").read_text())
    colors = ("green", "purple", "yellow")
    rgb = {"green": info["green_rgb"], "purple": [138 / 255, 126 / 255, 132 / 255],
           "yellow": [241 / 255, 217 / 255, 121 / 255]}
    centers = np.asarray(cfg["plates"]["centers_xy_m"]) + plate_offsets.reshape(3, 2)
    floor_z = info["floor_z_m"]
    physics = object_geom_attrs(physics_cfg, include_density=False)
    paths = []
    # Include order fixes plate keypoints to green=3, purple=4, yellow=5.
    for color in colors:
        xy = centers[plate_colors.index(color)]
        name = "target_plate" if color == cube_color else f"plate_{color}"
        root = ET.Element("mujoco", model=name)
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", name=f"{name}_visual", file=str(asset_dir / "plate_visual_0.obj"))
        texture_file = "green_0.png" if color == "green" else "plate_detail.png"
        ET.SubElement(asset, "texture", name=f"{name}_texture", type="2d",
                      file=str(asset_dir / texture_file))
        plate_rgba = [1, 1, 1, 1] if color == "green" else [*rgb[color], 1]
        ET.SubElement(asset, "material", name=f"{name}_material", specular="0.25", shininess="0.25",
                      texture=f"{name}_texture", rgba=format_vec(plate_rgba))
        body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name=name, pos=format_vec([*xy, 0]))
        ET.SubElement(body, "geom", name=f"{name}_visual", type="mesh", mesh=f"{name}_visual",
                      material=f"{name}_material", contype="0", conaffinity="0", mass="0", group="2")
        for index in range(info["collision_parts"]):
            mesh_name = f"{name}_part_{index}"
            ET.SubElement(asset, "mesh", name=mesh_name,
                          file=str(asset_dir / "collision" / f"part_{index:03d}.stl"))
            ET.SubElement(body, "geom", name=mesh_name, type="mesh", mesh=mesh_name,
                          rgba="0 0 0 0", group="3", **physics)
        ET.SubElement(body, "site", name=f"plate_{color}_center", pos=format_vec([0, 0, floor_z]), size="0.001", rgba="0 0 0 0")
        path = sample_dir / f"{name}.xml"
        ET.ElementTree(root).write(path, encoding="unicode")
        paths.append(path)

    half = cfg["cube"]["side_m"] / 2
    root = ET.Element("mujoco", model="cube")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="cube_visual", file=str(asset_dir / "cube.obj"))
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="cube",
                         pos=format_vec([cube_pose[0], cube_pose[1], half + .0005]),
                         quat=format_vec(yaw_deg_to_quat_wxyz(cube_pose[2])))
    ET.SubElement(body, "freejoint", name="cube_joint")
    ET.SubElement(body, "geom", name="cube_visual", type="mesh", mesh="cube_visual", group="2",
                  contype="0", conaffinity="0", mass="0", rgba=format_vec([*rgb[cube_color], 1]))
    ET.SubElement(body, "geom", name="cube_collision", type="box", size=format_vec([half] * 3),
                  mass=str(cfg["cube"]["mass_kg"]), rgba="0 0 0 0", group="3", **physics)
    sites = [("cube_center", [0, 0, 0]), ("cube_right", [0, -half, 0])]
    for name, pos in sites:
        ET.SubElement(body, "site", name=name, pos=format_vec(pos), size="0.001", rgba="0 0 0 0")
    cube_path = sample_dir / "cube.xml"
    ET.ElementTree(root).write(cube_path, encoding="unicode")
    paths.insert(0, cube_path)
    scene_path = write_base_scene_xml(sample_dir, task_name, "500M", gripper_xml_path, paths, physics_cfg)
    scene = ET.parse(scene_path)
    custom = ET.SubElement(scene.getroot(), "custom")
    target_keypoint = 3 + colors.index(cube_color)
    ET.SubElement(custom, "numeric", name="target_plate_keypoint", data=str(target_keypoint))
    scene.write(scene_path, encoding="unicode")
    (sample_dir / f"{task_name}_case.json").write_text(json.dumps({
        "cube_color": cube_color, "plate_colors": plate_colors,
        "plate_centers_xy_m": centers.tolist(), "target_slot": plate_colors.index(cube_color),
        "floor_z_m": floor_z,
        "target_plate_keypoint": target_keypoint,
    }, indent=2) + "\n")


class ArrangeCubeByColorSceneGenerator(AbstractTaskSceneGenerator):
    """Fixed green, purple, yellow plate order with position jitter."""

    def setup_bounds(self) -> np.ndarray:
        cube = self.cfg["cube"]
        jitter = self.cfg["plates"]["position_jitter_m"]
        return np.array([[0, 3], cube["x"], cube["y"], cube["yaw_deg"],
                         *[[-jitter, jitter]] * 6], dtype=float)

    def write_sample(self, sample_dir: Path, sample_params: np.ndarray) -> None:
        colors = ("green", "purple", "yellow")
        write_arrange_cube_by_color_scene(
            sample_dir, self.cfg, self.physics_cfg, self.gripper_xml_path,
            "arrange_cube_by_color", colors[int(sample_params[0])], colors,
            sample_params[1:4], sample_params[4:],
        )


def build_task_scene_generator(
    task_name: str,
    cfg: dict,
    gripper_xml_path: str | Path,
    object_dir: str | Path,
    **kwargs,
) -> AbstractTaskSceneGenerator:
    if task_name not in TASK_SCENE_GENERATORS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_SCENE_GENERATORS[task_name](cfg, gripper_xml_path, object_dir, **kwargs)

TASK_SCENE_GENERATORS = {"arrange_cube_by_color": ArrangeCubeByColorSceneGenerator}

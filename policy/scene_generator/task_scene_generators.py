from __future__ import annotations

import json
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

from keypoint_program_solver.utils import camera_element

import numpy as np
import trimesh

from .utils.default_world import write_scene_xml as write_base_scene_xml

from .utils.physics_config import (
    finger_geom_attrs,
    normalize_physics_config,
    object_density,
    object_geom_attrs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHOE_POSE_KEYS = ("x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg")
OBJECT_COLORS = {
    "red": "1 0 0 1",
    "green": "0.6 0.8 0.0 1",
    "blue": "0.4 0.6 0.7 1",
    "grey": "0.5 0.5 0.5 1",
    "brown": "0.6 0.3 0.1 1",
}


def yaw_deg_to_quat_wxyz(yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=float)


def axis_angle_quat_wxyz(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array(
        [
            np.cos(angle / 2.0),
            axis[0] * np.sin(angle / 2.0),
            axis[1] * np.sin(angle / 2.0),
            axis[2] * np.sin(angle / 2.0),
        ],
        dtype=float,
    )


def quat_multiply(q1: np.ndarray | list[float], q2: np.ndarray | list[float]) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def euler_xyz_deg_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    qx = axis_angle_quat_wxyz(np.array([1.0, 0.0, 0.0]), roll_deg)
    qy = axis_angle_quat_wxyz(np.array([0.0, 1.0, 0.0]), pitch_deg)
    qz = yaw_deg_to_quat_wxyz(yaw_deg)
    return quat_multiply(qz, quat_multiply(qy, qx))


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


def write_repo_object_xml(
    sample_dir: Path,
    object_dir: Path,
    category: str,
    instance_id: str,
    pos: np.ndarray,
    quat: np.ndarray,
    scale: np.ndarray,
    density: float | None,
    color: str,
    physics_cfg: dict,
) -> Path:
    source_prefix = f"{category}_{instance_id}"
    object_instance_dir = object_dir / category / instance_id
    root = ET.parse(object_instance_dir / f"{source_prefix}_decomposed_template.xml").getroot()
    with (object_instance_dir / f"{source_prefix}.json").open("r", encoding="utf-8") as file:
        annotation = json.load(file)
        keypoints = annotation["keypoints"]

    scale_str = format_vec(scale)
    for mesh in root.findall("./asset/mesh"):
        mesh.attrib["scale"] = scale_str
        mesh.attrib["file"] = str((object_instance_dir / mesh.attrib["file"]).resolve())
    for texture in root.findall("./asset/texture[@file]"):
        texture.set("file", str((object_instance_dir / texture.attrib["file"]).resolve()))

    body = root.findall("./worldbody/body")[0]
    body.set("pos", format_vec(pos))
    body.set("quat", format_vec(quat))
    body.insert(0, ET.Element("freejoint", name=f"{source_prefix}_freejoint"))

    if "mass" in physics_cfg["object"] or "mass_properties" in annotation:
        properties = annotation["mass_properties"]
        mass = (
            float(physics_cfg["object"]["mass"])
            if "mass" in physics_cfg["object"]
            else float(density * properties["volume"] * np.prod(scale))
        )
        if mass <= 0:
            raise ValueError("Object mass must be positive")
        unit_inertia = np.array(properties["inertia_per_kg"], dtype=float)
        covariance = np.trace(unit_inertia) / 2 * np.eye(3) - unit_inertia
        scaled_covariance = covariance * np.outer(scale, scale)
        inertia = mass * (np.trace(scaled_covariance) * np.eye(3) - scaled_covariance)
        ET.SubElement(
            body, "inertial", mass=str(mass),
            pos=format_vec(np.array(properties["center_of_mass"]) * scale),
            fullinertia=format_vec(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]),
        )

    for keypoint_id, keypoint in enumerate(keypoints):
        keypoint_pos = np.array(keypoint["position"], dtype=float) * scale
        ET.SubElement(
            body,
            "site",
            name=f"{source_prefix}_kp_{keypoint_id}",
            pos=format_vec(keypoint_pos),
            size="0.01",
            rgba="0 0 0 0",
            type="sphere",
        )

    geom_attrs = object_geom_attrs(physics_cfg, include_density=False)
    for geom in root.findall(".//geom"):
        if "mesh" in geom.attrib:
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue
            geom.set("type", "mesh")
            if "rgba" not in geom.attrib:
                geom.set("rgba", OBJECT_COLORS[color])
            for attr_name, attr_value in geom_attrs.items():
                geom.set(attr_name, attr_value)
            if density is None:
                geom.set("mass", "1")
            else:
                geom.set("density", str(float(density)))
                geom.attrib.pop("mass", None)

    ET.indent(root, space="  ")
    xml_path = sample_dir / f"{source_prefix}.xml"
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    return xml_path


def shoe_yaw_range(shoe_cfg: dict) -> list[float]:
    return shoe_yaw_intervals(shoe_cfg)[0].tolist()


def shoe_pose_bounds(shoe_cfg: dict) -> np.ndarray:
    return np.array(
        [
            shoe_cfg["x"],
            shoe_cfg["y"],
            shoe_cfg["z"],
            shoe_cfg["roll_deg"],
            shoe_cfg["pitch_deg"],
            shoe_yaw_range(shoe_cfg),
        ],
        dtype=float,
    )


def shoe_yaw_intervals(shoe_cfg: dict) -> np.ndarray:
    if "yaw_deg" in shoe_cfg:
        yaw_cfg = shoe_cfg["yaw_deg"]
    else:
        yaw_cfg = shoe_cfg["theta_deg"]

    if isinstance(yaw_cfg, str):
        return np.array([json.loads(interval.strip()) for interval in yaw_cfg.split(" U ")], dtype=float)

    if isinstance(yaw_cfg[0], list):
        return np.array(yaw_cfg, dtype=float)

    return np.array([yaw_cfg], dtype=float)


def sample_shoe_pose_params(rng: np.random.Generator, shoe_cfg: dict, n_samples: int) -> np.ndarray:
    bounds = shoe_pose_bounds(shoe_cfg)
    samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, len(SHOE_POSE_KEYS)))
    yaw_intervals = shoe_yaw_intervals(shoe_cfg)
    interval_ids = rng.integers(0, len(yaw_intervals), size=n_samples)
    samples[:, 5] = rng.uniform(yaw_intervals[interval_ids, 0], yaw_intervals[interval_ids, 1])
    return samples


def load_shoe_keypoints(keypoints_path: Path) -> dict[str, np.ndarray]:
    with keypoints_path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)

    positions = [np.array(keypoint["position"], dtype=float) for keypoint in annotation["keypoints"]]
    return {f"{i + 1:03d}_opening_{i + 1}": position for i, position in enumerate(positions)}


def shoe_collision_part_paths(collision_instance_dir: Path) -> list[Path]:
    object_name = collision_instance_dir.parent.name
    instance_id = collision_instance_dir.name
    decomposed_dir = collision_instance_dir / "decomposed"
    part_paths = list(decomposed_dir.glob(f"{object_name}_{instance_id}_decomposed_*.stl"))
    return sorted(part_paths, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


@dataclass(frozen=True)
class MujocoMeshAsset:
    mesh_name: str
    material_name: str
    obj_path: Path
    rgba: tuple
    texture_path: Path


def shoe_visual_assets(collision_instance_dir: Path, asset_prefix: str) -> list[MujocoMeshAsset]:
    source_prefix = f"{collision_instance_dir.parent.name}_{collision_instance_dir.name}"
    visual_asset_dir = collision_instance_dir / "visual_assets"
    obj_paths = sorted(
        visual_asset_dir.glob(f"{source_prefix}_visual_*.obj"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not obj_paths:
        raise FileNotFoundError(f"no visual OBJ assets found in {visual_asset_dir}")

    texture_path = (visual_asset_dir / f"{source_prefix}_texture_0.png").resolve()
    return [
        MujocoMeshAsset(
            mesh_name=f"{asset_prefix}_visual_mesh_{visual_id}",
            material_name=f"{asset_prefix}_material_{visual_id}",
            obj_path=obj_path.resolve(),
            rgba=(1.0, 1.0, 1.0, 1.0),
            texture_path=texture_path,
        )
        for visual_id, obj_path in enumerate(obj_paths)
    ]


def write_shoe_xml(
    sample_dir: Path,
    collision_instance_dir: Path,
    pos: np.ndarray,
    quat: np.ndarray,
    density: float,
    keypoints: dict[str, np.ndarray],
    physics_cfg: dict,
    object_name: str,
) -> Path:
    assets = shoe_visual_assets(collision_instance_dir, object_name)
    collision_part_paths = shoe_collision_part_paths(collision_instance_dir)
    if not collision_part_paths:
        raise FileNotFoundError(f"no CoACD collision parts found in {collision_instance_dir / 'decomposed'}")

    root = ET.Element("mujoco", model=object_name)
    asset_root = ET.SubElement(root, "asset")
    for asset in assets:
        ET.SubElement(
            asset_root,
            "mesh",
            name=asset.mesh_name,
            file=str(asset.obj_path),
        )
        texture_name = f"{asset.material_name}_texture"
        ET.SubElement(
            asset_root,
            "texture",
            name=texture_name,
            type="2d",
            file=str(asset.texture_path),
        )
        ET.SubElement(
            asset_root,
            "material",
            name=asset.material_name,
            rgba=format_vec(asset.rgba),
            texture=texture_name,
        )

    for part_id, part_path in enumerate(collision_part_paths):
        ET.SubElement(
            asset_root,
            "mesh",
            name=f"{object_name}_collision_mesh_{part_id}",
            file=str(part_path.resolve()),
        )

    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", name=object_name, pos=format_vec(pos), quat=format_vec(quat))
    ET.SubElement(body, "freejoint", name=f"{object_name}_freejoint")
    insole_path = collision_instance_dir / "visual_assets" / "yellow_insole.obj"
    if insole_path.is_file():
        ET.SubElement(
            asset_root, "mesh", name=f"{object_name}_insole_mesh",
            file=str(insole_path.resolve()),
        )
        ET.SubElement(
            body, "geom", name=f"{object_name}_insole", type="mesh",
            mesh=f"{object_name}_insole_mesh", rgba="1 0.72 0.015 1",
            mass="0", contype="0", conaffinity="0", group="1",
        )
    for asset_id, asset in enumerate(assets):
        ET.SubElement(
            body,
            "geom",
            name=f"{object_name}_visual_{asset_id}",
            type="mesh",
            mass="0",
            mesh=asset.mesh_name,
            material=asset.material_name,
            contype="0",
            conaffinity="0",
            group="1",
        )

    geom_attrs = object_geom_attrs(physics_cfg, include_density=True, density=density)
    if "mass" in physics_cfg["object"]:
        mass = float(physics_cfg["object"]["mass"])
        if mass <= 0:
            raise ValueError("Object mass must be positive")
        part_volumes = np.array([trimesh.load_mesh(path).volume for path in collision_part_paths])
        part_masses = mass * part_volumes / part_volumes.sum()
        geom_attrs.pop("density")
    for part_id in range(len(collision_part_paths)):
        geom = ET.SubElement(
            body,
            "geom",
            name=f"{object_name}_collision_{part_id}",
            type="mesh",
            mesh=f"{object_name}_collision_mesh_{part_id}",
            rgba="0 0 0 0",
            **geom_attrs,
        )
        if "mass" in physics_cfg["object"]:
            geom.set("mass", str(part_masses[part_id]))

    for name, keypoint_pos in keypoints.items():
        ET.SubElement(
            body,
            "site",
            name=f"{object_name}_{name}",
            pos=format_vec(keypoint_pos),
            size="0.01",
            rgba="0 0 0 0",
            type="sphere",
        )

    ET.indent(root, space="  ")
    xml_path = sample_dir / f"{object_name}.xml"
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    return xml_path


def write_flip_then_arrange_shoe_scene_xml(
    sample_dir: Path,
    gripper_xml_path: Path,
    shoe_xml_paths: list[Path],
    physics_cfg: dict,
) -> Path:
    path = write_base_scene_xml(
        sample_dir,
        "flip_then_arrange_shoe_scene",
        "256M",
        gripper_xml_path,
        shoe_xml_paths,
        physics_cfg=physics_cfg,
    )
    root = ET.parse(path).getroot()
    custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'numeric', name='shoe_tray_placement', data='1')
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='unicode')
    return path


def write_shoe_tray_xml(sample_dir: Path, tray_cfg: dict, physics_cfg: dict) -> Path:
    folder = resolve_repo_path(tray_cfg['asset_dir'])
    info = json.loads((folder / 'asset_info.json').read_text())
    root = ET.Element('mujoco', model='shoe_tray')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'mesh', name='shoe_tray_visual', file=str(folder / 'tray_visual.obj'))
    ET.SubElement(asset, 'texture', name='shoe_tray_texture', type='2d', file=str(folder / 'texture.png'))
    ET.SubElement(asset, 'material', name='shoe_tray_material', texture='shoe_tray_texture')
    body = ET.SubElement(ET.SubElement(root, 'worldbody'), 'body', name='shoe_tray',
                         pos=format_vec(tray_cfg['position_m']),
                         quat=format_vec(euler_xyz_deg_to_quat_wxyz(0, 0, tray_cfg['yaw_deg'])))
    ET.SubElement(body, 'geom', name='shoe_tray_visual', type='mesh', mesh='shoe_tray_visual',
                  material='shoe_tray_material', mass='0', contype='0', conaffinity='0', group='1')
    attrs = object_geom_attrs(physics_cfg, include_density=True, density=100.)
    ET.SubElement(body, 'geom', name='shoe_tray_floor', type='box', pos='0 0 .007',
                  size='.1725 .15 .007', rgba='0 0 0 0', **attrs)
    for index, filename in enumerate(info['walls']):
        name = f'shoe_tray_wall_{index}'
        ET.SubElement(asset, 'mesh', name=name, file=str(folder / filename))
        ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name, rgba='0 0 0 0', **attrs)
    right = np.array(info['slots_local_m']['right'])
    sites = [right, right + [.05, 0, 0], right + [0, 0, .05], info['slots_local_m']['left']]
    for index, point in enumerate(sites, 1):
        ET.SubElement(body, 'site', name=f'shoe_tray_{index:03d}', pos=format_vec(point), size='.004', rgba='0 0 0 0')
    ET.indent(root, space='  ')
    path = sample_dir / 'shoe_tray.xml'
    ET.ElementTree(root).write(path, encoding='unicode')
    return path


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

    def generate_samples(self, base_dir: str | Path, n_samples: int, seed: int) -> None:
        base_dir = Path(base_dir).expanduser().resolve()
        base_dir.mkdir(parents=True)

        for sample_id, sample_params in enumerate(self.sample_params(n_samples, seed)):
            self.generate_scene(sample_params, base_dir, sample_id)


def write_object_table_contacts(scene_path: Path, object_paths: list[Path], contact_cfg: dict) -> None:
    """Append explicit tabletop contacts for each object's collision geoms."""
    scene = ET.parse(scene_path)
    contact = scene.getroot().find("contact")
    if contact is None:
        contact = ET.SubElement(scene.getroot(), "contact")
    for object_path in object_paths:
        obj = ET.parse(object_path)
        for geom in obj.findall(".//geom"):
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue
            ET.SubElement(
                contact, "pair", name=f"{geom.attrib['name']}_table",
                geom1=geom.attrib["name"], geom2="tabletop",
                solref=format_vec(contact_cfg["solref"]),
                solimp=format_vec(contact_cfg["solimp"]),
                friction=format_vec(contact_cfg["friction"]),
                condim=str(contact_cfg["condim"]),
            )
    scene.write(scene_path, encoding="unicode")


class HangMugSceneGenerator(AbstractTaskSceneGenerator):
    def setup_bounds(self) -> np.ndarray:
        mug_ranges = self.cfg["mug"]
        rack_ranges = self.cfg["rack"]
        return np.array(
            [
                mug_ranges["x"],
                mug_ranges["y"],
                mug_ranges["theta_deg"],
                rack_ranges["x"],
                rack_ranges["y"],
                rack_ranges["theta_deg"],
            ],
            dtype=float,
        )

    def write_sample(self, sample_dir: Path, sample_params: np.ndarray) -> None:
        ax, ay, ath, bx, by, bth = sample_params
        base_mug = np.array(self.cfg["base_mug"], dtype=float)
        base_rack = np.array(self.cfg["base_rack"], dtype=float)
        mug_quat_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        rack_quat_init = np.array([-0.707107, 0.0, 0.0, 0.707107], dtype=float)

        mug_physics = {
            **self.physics_cfg,
            "object": {**self.physics_cfg["object"], **self.physics_cfg["mug"]},
        }

        mug_xml_path = write_repo_object_xml(
            sample_dir=sample_dir,
            object_dir=self.object_dir,
            category="039_mug",
            instance_id="1",
            pos=base_mug + np.array([ax, ay, 0.0], dtype=float),
            quat=quat_multiply(mug_quat_init, yaw_deg_to_quat_wxyz(ath)),
            scale=np.array([1.0, 1.0, 1.0], dtype=float),
            density=object_density(mug_physics),
            color="green",
            physics_cfg=mug_physics,
        )
        rack_xml_path = write_repo_object_xml(
            sample_dir=sample_dir,
            object_dir=self.object_dir,
            category="040_rack",
            instance_id="1",
            pos=base_rack + np.array([bx, by, 0.0], dtype=float),
            quat=quat_multiply(rack_quat_init, yaw_deg_to_quat_wxyz(bth)),
            scale=np.array([1.0, 1.0, 1.0], dtype=float),
            density=None,
            color="brown",
            physics_cfg=self.physics_cfg,
        )
        gripper_xml_path = write_gripper_xml(sample_dir, self.gripper_xml_path, self.physics_cfg)
        write_base_scene_xml(
            sample_dir,
            "scene",
            "64M",
            gripper_xml_path,
            [mug_xml_path, rack_xml_path],
            physics_cfg=self.physics_cfg,
        )
        write_object_table_contacts(
            sample_dir / "tmp_scene.xml", [mug_xml_path], self.physics_cfg["mug_table"],
        )


class ArrangeTableSceneGenerator(AbstractTaskSceneGenerator):
    def setup_bounds(self) -> np.ndarray:
        bowl_range = self.cfg["bowl"]
        bowl_position = bowl_range["placements"][0] if "placements" in bowl_range else bowl_range
        fork_range = self.cfg["fork"]
        spoon_range = self.cfg["spoon"]
        return np.array(
            [
                bowl_position["x"],
                bowl_position["y"],
                bowl_range["theta_deg"],
                fork_range["x"],
                fork_range["y"],
                fork_range["theta_deg"],
                spoon_range["x"],
                spoon_range["y"],
                spoon_range["theta_deg"],
            ],
            dtype=float,
        )

    def sample_params(self, n_samples: int, seed: int) -> np.ndarray:
        if "placements" not in self.cfg["bowl"]:
            return super().sample_params(n_samples, seed)
        placements = self.cfg["bowl"]["placements"]
        rng = np.random.default_rng(seed)
        # Draw each complete scene together so a seed's first scenes do not
        # depend on the requested batch size.
        draws = rng.random((n_samples, len(self.bounds) + 1))
        probabilities = np.array([placement["probability"] for placement in placements])
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Bowl placement probabilities must be nonnegative and sum to 1")
        choices = np.searchsorted(np.cumsum(probabilities), draws[:, 0], side="right")
        bowl_bounds = np.array([[placement["x"], placement["y"]] for placement in placements])[choices]
        bowl_xy = bowl_bounds[:, :, 0] + draws[:, 1:3] * (bowl_bounds[:, :, 1] - bowl_bounds[:, :, 0])
        remaining = self.bounds[2:, 0] + draws[:, 3:] * (self.bounds[2:, 1] - self.bounds[2:, 0])
        return np.column_stack((bowl_xy, remaining))

    def write_sample(self, sample_dir: Path, sample_params: np.ndarray) -> None:
        bx, by, bth, fx, fy, fth, sx, sy, sth = sample_params
        base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        bowl_physics = {**self.physics_cfg, "object": self.physics_cfg["bowl"]}
        fork_physics = {**self.physics_cfg, "object": self.physics_cfg["fork"]}
        spoon_physics = {**self.physics_cfg, "object": self.physics_cfg["spoon"]}

        bowl_xml_path = write_repo_object_xml(
            sample_dir=sample_dir,
            object_dir=self.object_dir,
            category="002_bowl",
            instance_id="8",
            pos=np.array([bx, by, self.cfg["bowl"]["z"] if "z" in self.cfg["bowl"] else 0.04], dtype=float),
            quat=quat_multiply(base_quat, yaw_deg_to_quat_wxyz(bth)),
            scale=np.array([0.14944, 0.14944, 0.14944], dtype=float),
            density=object_density(bowl_physics),
            color="blue",
            physics_cfg=bowl_physics,
        )
        fork_xml_path = write_repo_object_xml(
            sample_dir=sample_dir,
            object_dir=self.object_dir,
            category="033_fork",
            instance_id="3",
            pos=np.array([fx, fy, self.cfg["fork"]["z"] if "z" in self.cfg["fork"] else 0.01], dtype=float),
            quat=quat_multiply(base_quat, yaw_deg_to_quat_wxyz(fth)),
            # The measured v2 fork mesh and keypoints are already in meters.
            scale=np.array([1.0, 1.0, 1.0], dtype=float),
            density=object_density(fork_physics),
            color="blue",
            physics_cfg=fork_physics,
        )
        spoon_xml_path = write_repo_object_xml(
            sample_dir=sample_dir,
            object_dir=self.object_dir,
            category="034_spoon",
            instance_id="2",
            pos=np.array([sx, sy, self.cfg["spoon"]["z"] if "z" in self.cfg["spoon"] else 0.01], dtype=float),
            quat=quat_multiply(base_quat, yaw_deg_to_quat_wxyz(sth)),
            # The measured v2 spoon mesh and keypoints are already in meters.
            scale=np.array([1.0, 1.0, 1.0], dtype=float),
            density=object_density(spoon_physics),
            color="blue",
            physics_cfg=spoon_physics,
        )
        gripper_xml_path = write_gripper_xml(sample_dir, self.gripper_xml_path, self.physics_cfg)
        write_base_scene_xml(
            sample_dir,
            "scene",
            "64M",
            gripper_xml_path,
            [bowl_xml_path, fork_xml_path, spoon_xml_path],
            physics_cfg=self.physics_cfg,
        )
        write_object_table_contacts(
            sample_dir / "tmp_scene.xml",
            [bowl_xml_path],
            self.physics_cfg["object_table"],
        )
        write_object_table_contacts(
            sample_dir / "tmp_scene.xml",
            [fork_xml_path, spoon_xml_path],
            {**self.physics_cfg["object_table"], **self.physics_cfg["utensil_table"]},
        )


class FlipThenArrangeShoeSceneGenerator(AbstractTaskSceneGenerator):
    def __init__(
        self,
        cfg: dict,
        gripper_xml_path: str | Path,
        object_dir: str | Path,
        *,
        density: float = 100.0,
        physics_cfg: dict | None = None,
        cameras: dict,
    ) -> None:
        super().__init__(cfg, gripper_xml_path, object_dir, physics_cfg, cameras=cameras)
        self.density = object_density(self.physics_cfg, density)

    def setup_bounds(self) -> np.ndarray:
        yaws = [entry['yaw_deg'] for entry in self.cfg['tray']['orientations']]
        return np.vstack([shoe_pose_bounds(self.cfg['shoe_right']), [min(yaws), max(yaws)]])

    def sample_params(self, n_samples: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        shoes = sample_shoe_pose_params(rng, self.cfg['shoe_right'], n_samples)
        orientations = self.cfg['tray']['orientations']
        tray_yaws = rng.choice([entry['yaw_deg'] for entry in orientations], size=n_samples,
                              p=[entry['probability'] for entry in orientations])
        return np.column_stack([shoes, tray_yaws])

    def write_sample(self, sample_dir: Path, sample_params: np.ndarray) -> None:
        x, y, z, roll, pitch, right_yaw, tray_yaw = sample_params
        tray_cfg = {**self.cfg['tray'], 'yaw_deg': float(tray_yaw)}
        tray_info = json.loads((resolve_repo_path(tray_cfg['asset_dir']) / 'asset_info.json').read_text())
        yaw = np.deg2rad(tray_cfg['yaw_deg'])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        left_slot = np.array(tray_info['slots_local_m']['left'])
        left_pos = np.array(tray_cfg['position_m']) + rotation @ (left_slot + [0, 0, .0525])
        right_pos = np.array(self.cfg['shoe_right']['base']) + [x, y, z]
        poses = [
            ('shoe_right', right_pos, euler_xyz_deg_to_quat_wxyz(roll, pitch, right_yaw)),
            ('shoe_left', left_pos, euler_xyz_deg_to_quat_wxyz(-90, 0, tray_cfg['yaw_deg'] + 180)),
        ]
        paths = []
        for name, pos, quat in poses:
            shoe_cfg = self.cfg[name]
            paths.append(write_shoe_xml(
                sample_dir, resolve_repo_path(shoe_cfg['asset_dir']), pos, quat,
                self.density, load_shoe_keypoints(resolve_repo_path(shoe_cfg['keypoints_path'])),
                self.physics_cfg, name,
            ))
        paths.append(write_shoe_tray_xml(sample_dir, tray_cfg, self.physics_cfg))
        gripper = write_gripper_xml(sample_dir, self.gripper_xml_path, self.physics_cfg)
        write_flip_then_arrange_shoe_scene_xml(sample_dir, gripper, paths, self.physics_cfg)


TASK_SCENE_GENERATORS = {
    "hang_mug": HangMugSceneGenerator,
    "arrange_table": ArrangeTableSceneGenerator,
    "flip_then_arrange_shoe": FlipThenArrangeShoeSceneGenerator,
}

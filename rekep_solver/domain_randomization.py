from __future__ import annotations

import colorsys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from scene_generator.utils.runtime_scene import (
    RuntimeScene,
    load_scene_spec,
    runtime_scene_from_spec,
    save_scene_spec,
)
from rekep_solver.utils import _calc_cammatrices


TEXTURE_MAT_PREFIX = "__DR_MAT_"
TEXTURE_TEX_PREFIX = "__DR_TEX_"
ROBOT_ROOT_BODY = "link_base"
GROUND_BODY = "ground"
TABLE_BODY = "table"
TABLE_GEOM = "tabletop"
WALL_BODY = "wall"
WALL_GEOM = "white_wall"
LEGACY_TABLE_GEOM = "dr_tabletop"
LEGACY_FLOOR_GEOM = "dr_floor"
LEGACY_BACKGROUND_GEOM = "dr_background"
OBJECT_CATEGORY = "object"
RANDOMIZED_CATEGORIES = ("table", "wall", OBJECT_CATEGORY)
RANDOMIZED_CATEGORY_SET = frozenset(RANDOMIZED_CATEGORIES)
CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class TextureVariationConfig:
    random_base_color: bool
    color_span: float
    noise_std: float
    accent_span: int


@dataclass(frozen=True)
class PlaneRandomizationConfig:
    color_delta: float
    specular_range: tuple[float, float]
    shininess_range: tuple[float, float]
    texture: TextureVariationConfig


@dataclass(frozen=True)
class CameraExtrinsicRandomizationConfig:
    position_delta: tuple[float, float, float]
    rotation_delta_degrees: tuple[float, float, float]


@dataclass(frozen=True)
class CameraIntrinsicRandomizationConfig:
    focal_scale: tuple[float, float]


@dataclass(frozen=True)
class CameraRandomizationConfig:
    extrinsic: CameraExtrinsicRandomizationConfig
    intrinsic: CameraIntrinsicRandomizationConfig


@dataclass(frozen=True)
class TableGeometryRandomizationConfig:
    center_xy: tuple[float, float]
    width: float
    length: float
    size_delta: float


@dataclass(frozen=True)
class DomainRandomizationConfig:
    table: PlaneRandomizationConfig
    table_geometry: TableGeometryRandomizationConfig
    wall: PlaneRandomizationConfig
    object: PlaneRandomizationConfig
    camera: CameraRandomizationConfig
    cube_plate_colors: tuple[str, ...] = ()
    # Scale built-in lighting, texture base-color, and semantic HSV ranges about their midpoints.
    appearance_range_scale: float = 1.0


@dataclass(frozen=True)
class SharedDomainRandomizationConfig:
    table: PlaneRandomizationConfig
    table_geometry: TableGeometryRandomizationConfig
    wall: PlaneRandomizationConfig
    camera: CameraRandomizationConfig
    appearance_range_scale: float = 1.0


DEFAULT_DOMAIN_RANDOMIZATION_CONFIG = DomainRandomizationConfig(
    table_geometry=TableGeometryRandomizationConfig(
        center_xy=(0.2915, 0.013), width=0.767, length=1.534, size_delta=0.03,
    ),
    table=PlaneRandomizationConfig(
        color_delta=0.75,
        specular_range=(0.0, 1.0),
        shininess_range=(0.0, 1.0),
        texture=TextureVariationConfig(
            random_base_color=True,
            color_span=80.0,
            noise_std=50.0,
            accent_span=100,
        ),
    ),
    wall=PlaneRandomizationConfig(
        color_delta=0.75,
        specular_range=(0.0, 1.0),
        shininess_range=(0.0, 1.0),
        texture=TextureVariationConfig(
            random_base_color=True,
            color_span=80.0,
            noise_std=50.0,
            accent_span=100,
        ),
    ),
    object=PlaneRandomizationConfig(
        color_delta=0.06,
        specular_range=(0.1, 0.25),
        shininess_range=(0.1, 0.25),
        texture=TextureVariationConfig(
            random_base_color=False,
            color_span=8.0,
            noise_std=3.0,
            accent_span=5,
        ),
    ),
    camera=CameraRandomizationConfig(
        extrinsic=CameraExtrinsicRandomizationConfig(
            position_delta=(0.0, 0.0, 0.0),
            rotation_delta_degrees=(0.0, 0.0, 0.0),
        ),
        intrinsic=CameraIntrinsicRandomizationConfig(
            focal_scale=(1.0, 1.0),
        ),
    ),
)


def texture_variation_config_from_dict(config: dict[str, Any]) -> TextureVariationConfig:
    return TextureVariationConfig(
        random_base_color=bool(config["random_base_color"]),
        color_span=float(config["color_span"]),
        noise_std=float(config["noise_std"]),
        accent_span=int(config["accent_span"]),
    )


def plane_randomization_config_from_dict(config: dict[str, Any]) -> PlaneRandomizationConfig:
    return PlaneRandomizationConfig(
        color_delta=float(config["color_delta"]),
        specular_range=tuple(config["specular_range"]),
        shininess_range=tuple(config["shininess_range"]),
        texture=texture_variation_config_from_dict(config["texture"]),
    )


def camera_randomization_config_from_dict(config: dict[str, Any]) -> CameraRandomizationConfig:
    return CameraRandomizationConfig(
        extrinsic=CameraExtrinsicRandomizationConfig(
            position_delta=tuple(config["extrinsic"]["position_delta"]),
            rotation_delta_degrees=tuple(config["extrinsic"]["rotation_delta_degrees"]),
        ),
        intrinsic=CameraIntrinsicRandomizationConfig(
            focal_scale=tuple(config["intrinsic"]["focal_scale"]),
        ),
    )


def table_geometry_randomization_config_from_dict(config: dict[str, Any]) -> TableGeometryRandomizationConfig:
    return TableGeometryRandomizationConfig(
        center_xy=tuple(config["center_xy"]),
        width=float(config["width"]),
        length=float(config["length"]),
        size_delta=float(config["size_delta"]),
    )


def domain_randomization_config_from_dict(config: dict[str, Any]) -> DomainRandomizationConfig:
    return DomainRandomizationConfig(
        table=plane_randomization_config_from_dict(config["table"]),
        table_geometry=table_geometry_randomization_config_from_dict(config["table"]["geometry"]),
        wall=plane_randomization_config_from_dict(config["wall"]),
        object=plane_randomization_config_from_dict(config["object"]),
        camera=camera_randomization_config_from_dict(config["camera"]),
        appearance_range_scale=float(config.get("appearance_range_scale", 1.0)),
    )


def shared_domain_randomization_config_from_dict(
    config: dict[str, Any],
) -> SharedDomainRandomizationConfig:
    return SharedDomainRandomizationConfig(
        table=plane_randomization_config_from_dict(config["table"]),
        table_geometry=table_geometry_randomization_config_from_dict(config["table"]["geometry"]),
        wall=plane_randomization_config_from_dict(config["wall"]),
        camera=camera_randomization_config_from_dict(config["camera"]),
        appearance_range_scale=float(config.get("appearance_range_scale", 1.0)),
    )


def domain_randomization_config_for_task(
    cfg: SharedDomainRandomizationConfig,
    task_cfg: dict[str, Any],
) -> DomainRandomizationConfig:
    from rekep_solver.hang_mug_randomization import HangMugRandomizationConfig
    from rekep_solver.arrange_cube_randomization import ArrangeCubeRandomizationConfig
    from rekep_solver.pack_battery_randomization import PackBatteryRandomizationConfig
    if isinstance(cfg, (HangMugRandomizationConfig, ArrangeCubeRandomizationConfig,
                        PackBatteryRandomizationConfig)):
        return cfg
    return DomainRandomizationConfig(
        table=cfg.table,
        table_geometry=cfg.table_geometry,
        wall=cfg.wall,
        object=plane_randomization_config_from_dict(
            task_cfg["domain_randomization"]["object"]
        ),
        camera=cfg.camera,
        appearance_range_scale=cfg.appearance_range_scale,
        cube_plate_colors=tuple(task_cfg["domain_randomization"].get("cube_plate_colors", ())),
    )


def add_texture_slots(spec: mujoco.MjSpec) -> None:
    if spec.material(f"{TEXTURE_MAT_PREFIX}0__") is not None:
        return

    slot_count = max(len(spec.geoms), 1)
    rgb_role = mujoco.mjtTextureRole.mjTEXROLE_RGB

    for slot_idx in range(slot_count):
        texture_name = f"{TEXTURE_TEX_PREFIX}{slot_idx}__"
        material_name = f"{TEXTURE_MAT_PREFIX}{slot_idx}__"
        spec.add_texture(
            name=texture_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
        )
        texture = spec.texture(texture_name)
        texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
        texture.width = 64
        texture.height = 64
        texture.rgb1 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        texture.rgb2 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        material = spec.add_material(name=material_name)
        material.rgba = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        material.specular = 0.5
        material.shininess = 0.5
        material.textures[rgb_role] = texture_name


def geom_category(model: mujoco.MjModel, geom_id: int) -> str:
    geom_name = model.geom(geom_id).name
    if geom_name in (TABLE_GEOM, LEGACY_TABLE_GEOM):
        return "table"
    if geom_name == LEGACY_FLOOR_GEOM:
        return "floor"
    if geom_name == LEGACY_BACKGROUND_GEOM:
        return "background"
    if geom_name == WALL_GEOM:
        return "wall"

    body_id = int(model.geom_bodyid[geom_id])
    body_name = model.body(body_id).name
    if body_name == TABLE_BODY:
        return "table"
    if body_name == WALL_BODY:
        return "wall"
    if body_name == GROUND_BODY:
        return "floor"

    if int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_PLANE.value:
        return "floor"

    root_body_id = int(model.body_rootid[body_id])
    if model.body(root_body_id).name == ROBOT_ROOT_BODY:
        return "robot"

    return OBJECT_CATEGORY


def geom_visible_for_randomization(model: mujoco.MjModel, geom_id: int) -> bool:
    material_id = int(model.geom_matid[geom_id])
    if material_id >= 0:
        return bool(model.mat_rgba[material_id, 3] > 0.0)
    return bool(model.geom_rgba[geom_id, 3] > 0.0)


def randomizable_geom_category(model: mujoco.MjModel, geom_id: int) -> str | None:
    category = geom_category(model, geom_id)
    if category not in RANDOMIZED_CATEGORY_SET:
        return None
    if category == OBJECT_CATEGORY and not geom_visible_for_randomization(model, geom_id):
        return None
    return category


def randomization_config_for_category(
    cfg: DomainRandomizationConfig,
    category: str,
) -> PlaneRandomizationConfig:
    return {
        "table": cfg.table,
        "wall": cfg.wall,
        OBJECT_CATEGORY: cfg.object,
    }[category]


def material_category(categories: list[str]) -> str:
    return next(category for category in RANDOMIZED_CATEGORIES if category in categories)


def material_categories(model: mujoco.MjModel, material_id: int) -> list[str]:
    categories = []
    for geom_id in range(model.ngeom):
        if int(model.geom_matid[geom_id]) == material_id:
            category = randomizable_geom_category(model, geom_id)
            if category is None:
                return []
            categories.append(category)

    return sorted(set(categories))


def surface_color(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    material_id = int(model.geom_matid[geom_id])
    if material_id >= 0:
        return model.mat_rgba[material_id, :3].copy()

    return model.geom_rgba[geom_id, :3].copy()


def randomize_lighting(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.RandomState,
    range_scale: float = 1.0,
) -> None:
    for light_id in range(model.nlight):
        position = np.array(
            [
                rng.uniform(-2.5, 2.5),
                rng.uniform(-2.5, 2.5),
                rng.uniform(1.0, 5.5),
            ]
        )
        target = np.array(
            [
                rng.uniform(-0.5, 0.6),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.0, 1.0),
            ]
        )
        position = np.array([0.0, 0.0, 3.25]) + range_scale * (position - [0.0, 0.0, 3.25])
        target = np.array([0.05, 0.0, 0.5]) + range_scale * (target - [0.05, 0.0, 0.5])
        direction = target - position
        direction = direction / np.linalg.norm(direction)

        model.light_pos[light_id] = position
        model.light_dir[light_id] = direction
        model.light_diffuse[light_id] = 0.6 + range_scale * (rng.uniform(0.2, 1.0, size=3) - 0.6)
        model.light_ambient[light_id] = 0.25 + range_scale * (rng.uniform(0.0, 0.5, size=3) - 0.25)
        model.light_specular[light_id] = 0.5 + range_scale * (rng.uniform(0.0, 1.0, size=3) - 0.5)
        model.light_active[light_id] = 1

        if hasattr(model, "light_castshadow"):
            model.light_castshadow[light_id] = 1

    mujoco.mj_forward(model, data)


def original_material_ids(model: mujoco.MjModel) -> list[int]:
    material_ids = []
    for material_id in range(model.nmat):
        name = model.mat(material_id).name
        if not name.startswith(TEXTURE_MAT_PREFIX):
            material_ids.append(material_id)
    return material_ids


def randomize_materials(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.RandomState,
    cfg: DomainRandomizationConfig,
) -> None:
    for material_id in original_material_ids(model):
        old_rgba = model.mat_rgba[material_id].copy()
        categories = material_categories(model, material_id)
        if not categories:
            continue
        if not all(category in RANDOMIZED_CATEGORY_SET for category in categories):
            continue
        category = material_category(categories)
        category_cfg = randomization_config_for_category(cfg, category)

        rgba = old_rgba.copy()
        rgba[:3] = np.clip(
            rgba[:3] + rng.uniform(-category_cfg.color_delta, category_cfg.color_delta, size=3),
            0,
            1,
        )
        model.mat_rgba[material_id] = rgba
        model.mat_specular[material_id] = rng.uniform(*category_cfg.specular_range)
        model.mat_shininess[material_id] = rng.uniform(*category_cfg.shininess_range)

    for geom_id in range(model.ngeom):
        category = randomizable_geom_category(model, geom_id)
        if category is None:
            continue
        category_cfg = randomization_config_for_category(cfg, category)
        rgba = model.geom_rgba[geom_id].copy()
        rgba[:3] = np.clip(
            rgba[:3] + rng.uniform(-category_cfg.color_delta, category_cfg.color_delta, size=3),
            0,
            1,
        )
        model.geom_rgba[geom_id] = rgba

    mujoco.mj_forward(model, data)


def texture_slot_ids(model: mujoco.MjModel) -> list[tuple[int, int]]:
    slots = []
    texture_name_to_id = {
        model.tex(texture_id).name: texture_id
        for texture_id in range(model.ntex)
    }

    for material_id in range(model.nmat):
        material_name = model.mat(material_id).name
        if not material_name.startswith(TEXTURE_MAT_PREFIX):
            continue
        slot_number = material_name[len(TEXTURE_MAT_PREFIX) : -2]
        texture_name = f"{TEXTURE_TEX_PREFIX}{slot_number}__"
        slots.append((material_id, texture_name_to_id[texture_name]))

    return slots


def procedural_texture(
    height: int,
    width: int,
    channels: int,
    rng: np.random.RandomState,
    base_color: np.ndarray,
    color_span: float,
    noise_std: float,
    accent_span: int,
) -> np.ndarray:
    base_color = np.clip(
        base_color[:channels] * 255 + rng.uniform(-color_span, color_span, size=channels),
        0,
        255,
    )
    noise = rng.normal(0, noise_std, size=(height, width, channels))
    checker = ((np.indices((height, width)).sum(axis=0) // 8) % 2)[..., None]
    accent = rng.randint(-accent_span, accent_span + 1, size=channels)
    bitmap = base_color + noise + checker * accent
    return np.clip(bitmap, 0, 255).astype(np.uint8)


def resize_bitmap_nearest(bitmap: np.ndarray, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, bitmap.shape[0] - 1, height).astype(int)
    x_idx = np.linspace(0, bitmap.shape[1] - 1, width).astype(int)
    return bitmap[y_idx][:, x_idx]


def write_texture_bitmap(model: mujoco.MjModel, texture_id: int, bitmap: np.ndarray) -> None:
    height = int(model.tex_height[texture_id])
    width = int(model.tex_width[texture_id])
    channels = int(model.tex_nchannel[texture_id])
    texture_address = int(model.tex_adr[texture_id])
    texture_size = height * width * channels

    if bitmap.shape != (height, width, channels):
        bitmap = resize_bitmap_nearest(bitmap, height, width)

    model.tex_data[texture_address : texture_address + texture_size] = bitmap.reshape(-1)


def read_texture_bitmap(model: mujoco.MjModel, texture_id: int) -> np.ndarray:
    height = int(model.tex_height[texture_id])
    width = int(model.tex_width[texture_id])
    channels = int(model.tex_nchannel[texture_id])
    texture_address = int(model.tex_adr[texture_id])
    texture_size = height * width * channels
    return model.tex_data[
        texture_address : texture_address + texture_size
    ].reshape(height, width, channels).copy()


def perturb_existing_texture_bitmap(
    bitmap: np.ndarray,
    rng: np.random.RandomState,
    color_span: float,
    noise_std: float,
    accent_span: int,
) -> np.ndarray:
    channels = bitmap.shape[-1]
    color_shift = rng.uniform(-color_span, color_span, size=channels)
    noise = rng.normal(0, noise_std, size=bitmap.shape)
    checker = ((np.indices(bitmap.shape[:2]).sum(axis=0) // 8) % 2)[..., None]
    accent = rng.randint(-accent_span, accent_span + 1, size=channels)
    randomized = bitmap.astype(np.float64) + color_shift + noise + checker * accent
    randomized = np.clip(randomized, 0, 255).astype(np.uint8)
    if channels == 4:
        randomized[..., 3] = bitmap[..., 3]
    return randomized


def randomize_existing_object_texture(
    model: mujoco.MjModel,
    rng: np.random.RandomState,
    geom_id: int,
    rgb_role: int,
    texture_cfg: TextureVariationConfig,
    randomized_texture_ids: set[int],
) -> None:
    material_id = int(model.geom_matid[geom_id])
    if material_id < 0:
        return

    texture_id = int(model.mat_texid[material_id, rgb_role])
    if texture_id < 0 or texture_id in randomized_texture_ids:
        return

    bitmap = read_texture_bitmap(model, texture_id)
    write_texture_bitmap(
        model,
        texture_id,
        perturb_existing_texture_bitmap(
            bitmap,
            rng,
            texture_cfg.color_span,
            texture_cfg.noise_std,
            texture_cfg.accent_span,
        ),
    )
    randomized_texture_ids.add(texture_id)


def randomize_textures(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.RandomState,
    cfg: DomainRandomizationConfig,
) -> None:
    slots = texture_slot_ids(model)
    rgb_role = mujoco.mjtTextureRole.mjTEXROLE_RGB.value
    randomized_object_texture_ids: set[int] = set()

    for geom_id in range(model.ngeom):
        category = randomizable_geom_category(model, geom_id)
        if category is None:
            continue

        category_cfg = randomization_config_for_category(cfg, category)
        texture_cfg = category_cfg.texture
        if category == OBJECT_CATEGORY:
            randomize_existing_object_texture(
                model,
                rng,
                geom_id,
                rgb_role,
                texture_cfg,
                randomized_object_texture_ids,
            )
            continue

        material_id, texture_id = slots[geom_id % len(slots)]
        height = int(model.tex_height[texture_id])
        width = int(model.tex_width[texture_id])
        channels = int(model.tex_nchannel[texture_id])

        if texture_cfg.random_base_color:
            base_color = 0.5 + cfg.appearance_range_scale * (rng.uniform(0.0, 1.0, size=channels) - 0.5)
        else:
            base_color = surface_color(model, geom_id)

        bitmap = procedural_texture(
            height,
            width,
            channels,
            rng,
            base_color,
            texture_cfg.color_span,
            texture_cfg.noise_std,
            texture_cfg.accent_span,
        )

        model.geom_matid[geom_id] = material_id
        model.mat_texid[material_id, rgb_role] = texture_id
        model.mat_rgba[material_id] = np.array([1.0, 1.0, 1.0, 1.0])
        model.mat_specular[material_id] = rng.uniform(*category_cfg.specular_range)
        model.mat_shininess[material_id] = rng.uniform(*category_cfg.shininess_range)

        write_texture_bitmap(model, texture_id, bitmap)

    mujoco.mj_forward(model, data)


def axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    return quat


def camera_rotation_quat(delta_rpy_degrees: np.ndarray) -> np.ndarray:
    delta_rpy = np.deg2rad(delta_rpy_degrees)
    qx = axis_angle_quat(np.array([1.0, 0.0, 0.0], dtype=np.float64), delta_rpy[0])
    qy = axis_angle_quat(np.array([0.0, 1.0, 0.0], dtype=np.float64), delta_rpy[1])
    qz = axis_angle_quat(np.array([0.0, 0.0, 1.0], dtype=np.float64), delta_rpy[2])

    qzy = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(qzy, qz, qy)
    qzyx = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(qzyx, qzy, qx)
    return qzyx


def camera_delta_extrinsic_matrix(
    translation_delta: np.ndarray,
    rotation_delta_degrees: np.ndarray,
) -> np.ndarray:
    delta_quat = camera_rotation_quat(rotation_delta_degrees)
    delta_rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(delta_rotation, delta_quat)

    delta_extrinsic = np.eye(4, dtype=np.float64)
    delta_extrinsic[:3, :3] = delta_rotation.reshape(3, 3)
    delta_extrinsic[:3, 3] = translation_delta
    return delta_extrinsic


def set_camera_extrinsic_matrix(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    extrinsic: np.ndarray,
) -> None:
    camera_rotation = extrinsic[:3, :3]
    camera_position = -camera_rotation.T @ extrinsic[:3, 3]
    camera_world_rotation = camera_rotation.T @ CAMERA_AXIS_FLIP

    body_id = int(model.cam_bodyid[camera_id])
    body_world_rotation = data.xmat[body_id].reshape(3, 3)
    body_world_position = data.xpos[body_id]
    local_position = body_world_rotation.T @ (camera_position - body_world_position)
    local_rotation = body_world_rotation.T @ camera_world_rotation

    camera_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(camera_quat, local_rotation.reshape(-1))
    model.cam_pos[camera_id] = local_position
    model.cam_quat[camera_id] = camera_quat


def set_camera_intrinsic_matrix(
    model: mujoco.MjModel,
    camera_id: int,
    intrinsic: np.ndarray,
) -> None:
    model.cam_intrinsic[camera_id, 0] = (
        intrinsic[0, 0]
        / model.cam_resolution[camera_id, 0]
        * model.cam_sensorsize[camera_id, 0]
    )
    model.cam_intrinsic[camera_id, 1] = (
        intrinsic[1, 1]
        / model.cam_resolution[camera_id, 1]
        * model.cam_sensorsize[camera_id, 1]
    )


def randomize_cameras(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.RandomState,
    cfg: CameraRandomizationConfig,
) -> None:
    position_delta = np.asarray(cfg.extrinsic.position_delta, dtype=np.float64)
    rotation_delta = np.asarray(cfg.extrinsic.rotation_delta_degrees, dtype=np.float64)
    focal_scale = np.asarray(cfg.intrinsic.focal_scale, dtype=np.float64)

    for camera_id in range(model.ncam):
        camera_name = model.camera(camera_id).name
        extrinsic, intrinsic = _calc_cammatrices(data, camera_name, {})

        translation_delta = rng.uniform(-position_delta, position_delta)
        delta_rpy = rng.uniform(-rotation_delta, rotation_delta)
        delta_extrinsic = camera_delta_extrinsic_matrix(translation_delta, delta_rpy)
        randomized_extrinsic = delta_extrinsic @ extrinsic
        set_camera_extrinsic_matrix(model, data, camera_id, randomized_extrinsic)

        intrinsic_delta = np.diag(
            [
                rng.uniform(focal_scale[0], focal_scale[1]),
                rng.uniform(focal_scale[0], focal_scale[1]),
                1.0,
            ]
        )
        randomized_intrinsic = intrinsic.copy()
        randomized_intrinsic[:3, :] = randomized_intrinsic[:3, :] @ intrinsic_delta
        set_camera_intrinsic_matrix(model, camera_id, randomized_intrinsic)

    mujoco.mj_forward(model, data)


def randomize_table_geometry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.RandomState,
    cfg: TableGeometryRandomizationConfig,
) -> None:
    table = model.geom(TABLE_GEOM)
    size_xy = np.array([cfg.width, cfg.length]) + rng.uniform(-cfg.size_delta, cfg.size_delta, 2)
    table.size[:2] = size_xy / 2
    # The shared scene table is an axis-aligned box on a body at world origin.
    table.pos[:2] = cfg.center_xy
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)


def randomize_cube_plate_colors(
    model: mujoco.MjModel,
    rng: np.random.RandomState,
    colors: tuple[str, ...],
    range_scale: float = 1.0,
) -> None:
    """Recolor the three semantic plate groups and match the cube to its target.

    Site names preserve group identity even when a plate is named target_plate.
    Plate texture detail is retained as neutral brightness modulation.
    """
    hue_ranges = {"blue": (205 / 360, 235 / 360),
                  "purple": (285 / 360, 295 / 360),
                  "yellow": (48 / 360, 62 / 360),
                  "green": (100 / 360, 140 / 360)}
    rgb_role = mujoco.mjtTextureRole.mjTEXROLE_RGB.value
    for source_color, color in zip(("green", "purple", "yellow"), colors, strict=True):
        body_id = int(model.site(f"plate_{source_color}_center").bodyid[0])
        plate_name = model.body(body_id).name
        saturation_range = (.55, .65) if color == "purple" else (.45, .75)
        brightness_range = (.72, .82) if color == "purple" else (.65, .90)
        hsv = np.array([rng.uniform(*hue_ranges[color]),
                        rng.uniform(*saturation_range), rng.uniform(*brightness_range)])
        midpoint = np.array([np.mean(hue_ranges[color]), np.mean(saturation_range),
                             np.mean(brightness_range)])
        hsv = midpoint + range_scale * (hsv - midpoint)
        names = [f"{plate_name}_visual"]
        if plate_name == "target_plate":
            names.append("cube_visual")
        for name in names:
            # Independent small offsets keep matched objects close in color.
            variation = hsv + range_scale * rng.uniform([-2 / 360, -.025, -.025],
                                          [2 / 360, .025, .025])
            rgb = colorsys.hsv_to_rgb(*variation)
            geom = model.geom(name)
            material_id = int(geom.matid[0])
            if material_id < 0:
                geom.rgba[:3] = rgb
            else:
                model.mat_rgba[material_id, :3] = rgb
                texture_id = int(model.mat_texid[material_id, rgb_role])
                if texture_id >= 0:
                    bitmap = read_texture_bitmap(model, texture_id).copy()
                    # Remove the original hue so a green texture cannot tint blue.
                    brightness = bitmap[..., :3].mean(axis=-1, keepdims=True) / 255
                    bitmap[..., :3] = np.rint(255 * (.95 + .05 * brightness)).astype(np.uint8)
                    write_texture_bitmap(model, texture_id, bitmap)


def apply_domain_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.RandomState,
    cfg: DomainRandomizationConfig = DEFAULT_DOMAIN_RANDOMIZATION_CONFIG,
) -> None:
    randomize_table_geometry(model, data, rng, cfg.table_geometry)
    randomize_lighting(model, data, rng, cfg.appearance_range_scale)
    randomize_materials(model, data, rng, cfg)
    randomize_textures(model, data, rng, cfg)
    if cfg.cube_plate_colors:
        randomize_cube_plate_colors(model, rng, cfg.cube_plate_colors, cfg.appearance_range_scale)
    randomize_cameras(model, data, rng, cfg.camera)


def prepare_spec_for_domain_randomization(
    spec: mujoco.MjSpec,
    texture_output_dir: str | Path,
    cfg: DomainRandomizationConfig = DEFAULT_DOMAIN_RANDOMIZATION_CONFIG,
) -> mujoco.MjSpec:
    Path(texture_output_dir).mkdir(parents=True, exist_ok=True)
    add_texture_slots(spec)
    return spec


def domain_randomization_metadata(
    seed: int,
    cfg: DomainRandomizationConfig = DEFAULT_DOMAIN_RANDOMIZATION_CONFIG,
    texture_output_dir: str | Path | None = None,
    source_xml_path: str | Path | None = None,
    prepared_xml_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "seed": int(seed),
        "modes": ["lighting", "tabletop", "wall", "object_visual", "camera"],
        "config": asdict(cfg),
    }
    if texture_output_dir is not None:
        metadata["texture_output_dir"] = str(Path(texture_output_dir))
    if source_xml_path is not None:
        metadata["source_xml_path"] = str(Path(source_xml_path))
    if prepared_xml_path is not None:
        metadata["prepared_xml_path"] = str(Path(prepared_xml_path))
    return metadata


def build_domain_randomized_runtime_scene(
    scene_source: str | Path | mujoco.MjSpec,
    texture_output_dir: str | Path,
    seed: int,
    cfg: DomainRandomizationConfig = DEFAULT_DOMAIN_RANDOMIZATION_CONFIG,
    prepared_xml_path: str | Path | None = None,
) -> RuntimeScene:
    from rekep_solver.pack_battery_randomization import PackBatteryRandomizationConfig
    if isinstance(cfg, PackBatteryRandomizationConfig):
        from rekep_solver.pack_battery_randomization import build_runtime_scene
        return build_runtime_scene(scene_source, seed, cfg)
    from rekep_solver.arrange_cube_randomization import ArrangeCubeRandomizationConfig
    if isinstance(cfg, ArrangeCubeRandomizationConfig):
        from rekep_solver.arrange_cube_randomization import build_runtime_scene
        return build_runtime_scene(scene_source, seed, cfg)
    from rekep_solver.hang_mug_randomization import HangMugRandomizationConfig, build_runtime_scene
    if isinstance(cfg, HangMugRandomizationConfig):
        return build_runtime_scene(scene_source, seed, cfg)
    if isinstance(scene_source, mujoco.MjSpec):
        spec = scene_source
        source_xml_path = None
    else:
        source_xml_path = Path(scene_source).resolve()
        spec = load_scene_spec(source_xml_path)

    prepare_spec_for_domain_randomization(spec, texture_output_dir, cfg)
    saved_prepared_xml_path = None
    if prepared_xml_path is not None:
        saved_prepared_xml_path = save_scene_spec(spec, prepared_xml_path)

    scene = runtime_scene_from_spec(
        spec,
        source_xml_path=source_xml_path,
        metadata={
            "domain_randomization": domain_randomization_metadata(
                seed=seed,
                cfg=cfg,
                texture_output_dir=texture_output_dir,
                source_xml_path=source_xml_path,
                prepared_xml_path=saved_prepared_xml_path,
            )
        },
    )
    apply_domain_randomization(scene.model, scene.data, np.random.RandomState(seed), cfg)
    return scene


def build_domain_randomized_model(
    scene_source: str | Path | mujoco.MjSpec,
    texture_output_dir: str | Path,
    seed: int,
    cfg: DomainRandomizationConfig = DEFAULT_DOMAIN_RANDOMIZATION_CONFIG,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    scene = build_domain_randomized_runtime_scene(scene_source, texture_output_dir, seed, cfg)
    return scene.model, scene.data

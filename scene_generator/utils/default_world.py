from __future__ import annotations

from pathlib import Path

from .physics_config import ground_geom_attrs, normalize_physics_config, tabletop_geom_attrs, xml_attrs


TABLE_BODY_NAME = "table"
TABLE_GEOM_NAME = "tabletop"
WALL_BODY_NAME = "wall"
WALL_GEOM_NAME = "white_wall"
DEFAULT_TABLE_CENTER = (0.2135, -0.0595, -0.005)
DEFAULT_TABLE_SIZE = (0.747, 1.527, 0.01)
DEFAULT_TABLE_RGBA = (1.0, 1.0, 1.0, 1.0)
DEFAULT_WALL_CENTER = (DEFAULT_TABLE_CENTER[0] - DEFAULT_TABLE_SIZE[0] / 2 - 0.6 - 0.005, 0.0, -0.125)
DEFAULT_WALL_SIZE = (0.01, 3, 1.75)
DEFAULT_WALL_RGBA = (1.0, 1.0, 1.0, 1.0)
MATERIAL_DIR = Path(__file__).resolve().parents[2] / "assets/materials/table_arrange"
DEFAULT_TABLE_FRICTION = (0.1, 0.005, 0.0001)


def format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def half_size(values: tuple[float, float, float]) -> str:
    return format_vec(tuple(float(value) / 2.0 for value in values))


def default_worldbody_xml(physics_cfg: dict | None = None) -> str:
    physics = normalize_physics_config(physics_cfg)
    table_attrs = xml_attrs(tabletop_geom_attrs(physics), "                    ")
    ground_attrs = xml_attrs(ground_geom_attrs(physics), "                    ")
    return f'''<worldbody>
            <light diffuse="0.8 0.8 0.8"
                specular="0.2 0.2 0.2"
                pos="0 0 4"
                dir="0 0 -1"
                cutoff="180"
                castshadow="false"
                directional="true"/>

            <body name="{TABLE_BODY_NAME}" pos="0 0 0">
                <geom name="{TABLE_GEOM_NAME}"
                    type="box"
                    pos="{format_vec(DEFAULT_TABLE_CENTER)}"
                    size="{half_size(DEFAULT_TABLE_SIZE)}"
                    rgba="{format_vec(DEFAULT_TABLE_RGBA)}"
                    material="default_table_wood"
                    contype="1"
                    conaffinity="1"
                    {table_attrs}/>
            </body>

            <body name="{WALL_BODY_NAME}" pos="0 0 0">
                <geom name="{WALL_GEOM_NAME}"
                    type="box"
                    pos="{format_vec(DEFAULT_WALL_CENTER)}"
                    size="{half_size(DEFAULT_WALL_SIZE)}"
                    rgba="{format_vec(DEFAULT_WALL_RGBA)}"
                    material="default_white_wall"
                    contype="0"
                    conaffinity="0"/>
            </body>

            <geom name="carpet_floor" type="plane" pos="0 0 -1" size="0 0 0.1"
                material="default_carpet" {ground_attrs}/>
        </worldbody>'''


def scene_xml(
    model_name: str,
    memory: str,
    gripper_xml_path: str | Path,
    object_xml_paths: list[str | Path],
    physics_cfg: dict | None = None,
) -> str:
    object_includes = "\n".join(f'  <include file="{Path(path).resolve()}"/>' for path in object_xml_paths)
    return f'''<mujoco model="{model_name}">
  <size memory="{memory}"/>
  <include file="{Path(gripper_xml_path).resolve()}"/>
{object_includes}
  <asset>
    <texture name="default_table_wood" type="2d" file="{MATERIAL_DIR / 'wood_e68650.png'}"/>
    <material name="default_table_wood" texture="default_table_wood" texrepeat="1 1"
              texuniform="false" specular="0.1" shininess="0.15"/>
    <texture name="default_carpet" type="2d" file="{MATERIAL_DIR / 'carpet_gray_woven.png'}"/>
    <material name="default_carpet" texture="default_carpet" texrepeat="2 2"
              texuniform="true" specular="0" shininess="0" reflectance="0"/>
    <material name="default_white_wall" rgba="1 1 1 1" specular="0" emission="0.4"/>
  </asset>
  {default_worldbody_xml(physics_cfg)}
</mujoco>
'''


def write_scene_xml(
    sample_dir: str | Path,
    model_name: str,
    memory: str,
    gripper_xml_path: str | Path,
    object_xml_paths: list[str | Path],
    physics_cfg: dict | None = None,
) -> Path:
    scene_xml_path = Path(sample_dir) / "tmp_scene.xml"
    scene_xml_path.write_text(
        scene_xml(model_name, memory, gripper_xml_path, object_xml_paths, physics_cfg),
        encoding="utf-8",
    )
    return scene_xml_path

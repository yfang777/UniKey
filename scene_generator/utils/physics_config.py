from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PHYSICS_CONFIG = {
    "solver": {
        "impratio": 5,
        "noslip_iterations": 5,
    },
    "ground": {
        "friction": [0.1, 0.005, 0.0001],
    },
    "tabletop": {
        "friction": [0.1, 0.005, 0.0001],
    },
    "object": {
        "friction": [10.0, 5.0, 5.0],
        "condim": 6,
        "density": 100.0,
    },
    "finger": {
        "friction": [20.0, 10.0, 10.0],
        "condim": 6,
    },
    "sticky": {
        "enabled": False,
        "friction": [10.0, 5.0, 5.0],
        "condim": 6,
    },
}

GEOM_ATTR_KEYS = ("friction", "condim", "solref", "solimp", "margin", "gap", "priority", "density")


def normalize_physics_config(physics_cfg: dict[str, Any] | None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_PHYSICS_CONFIG)
    if physics_cfg is None:
        return config

    for section_name, section_values in physics_cfg.items():
        if section_name not in config:
            config[section_name] = deepcopy(section_values)
        elif isinstance(section_values, dict):
            config[section_name].update(deepcopy(section_values))
        else:
            config[section_name] = deepcopy(section_values)
    return config


def load_task_physics_config(task_name: str, tasks_config_path: str | Path) -> dict[str, Any]:
    with Path(tasks_config_path).open("r", encoding="utf-8") as file:
        tasks_cfg = yaml.safe_load(file)["tasks"]
    if task_name not in tasks_cfg:
        return normalize_physics_config(None)
    task_cfg = tasks_cfg[task_name]
    physics_cfg = task_cfg["physics"] if "physics" in task_cfg else None
    return normalize_physics_config(physics_cfg)


def format_mujoco_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(float(item)) for item in value)
    return str(value)


def geom_attrs(section_cfg: dict[str, Any]) -> dict[str, str]:
    return {
        key: format_mujoco_value(section_cfg[key])
        for key in GEOM_ATTR_KEYS
        if key in section_cfg
    }


def xml_attrs(attrs: dict[str, str], indent: str = "") -> str:
    return f"\n{indent}".join(
        f'{attr_name}="{attr_value}"'
        for attr_name, attr_value in attrs.items()
    )


def object_section(physics_cfg: dict[str, Any]) -> dict[str, Any]:
    config = normalize_physics_config(physics_cfg)
    section = deepcopy(config["object"])
    sticky = config["sticky"]
    if sticky["enabled"]:
        section.update({key: value for key, value in sticky.items() if key != "enabled"})
    return section


def object_density(physics_cfg: dict[str, Any], fallback: float = 100.0) -> float:
    section = object_section(physics_cfg)
    return float(section["density"] if "density" in section else fallback)


def object_geom_attrs(
    physics_cfg: dict[str, Any],
    *,
    include_density: bool,
    density: float | None = None,
) -> dict[str, str]:
    section = object_section(physics_cfg)
    if not include_density:
        section.pop("density", None)
    elif density is not None:
        section["density"] = float(density)
    return geom_attrs(section)


def ground_geom_attrs(physics_cfg: dict[str, Any]) -> dict[str, str]:
    config = normalize_physics_config(physics_cfg)
    return geom_attrs(config["ground"])


def tabletop_geom_attrs(physics_cfg: dict[str, Any]) -> dict[str, str]:
    config = normalize_physics_config(physics_cfg)
    return geom_attrs(config["tabletop"])


def finger_geom_attrs(physics_cfg: dict[str, Any]) -> dict[str, str]:
    config = normalize_physics_config(physics_cfg)
    return geom_attrs(config["finger"])

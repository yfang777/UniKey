from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco

from robot.model import scene_model_text


@dataclass
class RuntimeScene:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    data: mujoco.MjData
    source_xml_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_scene_spec(xml_path: str | Path) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(xml_path))


def compile_scene_spec(spec: mujoco.MjSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def runtime_scene_from_spec(
    spec: mujoco.MjSpec,
    source_xml_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> RuntimeScene:
    model, data = compile_scene_spec(spec)
    source_path = None if source_xml_path is None else Path(source_xml_path).resolve()
    scene_metadata = {} if metadata is None else dict(metadata)
    scene_metadata["robot_variant"] = scene_model_text(model, "robot_variant")
    return RuntimeScene(
        spec=spec,
        model=model,
        data=data,
        source_xml_path=source_path,
        metadata=scene_metadata,
    )


def runtime_scene_from_xml(xml_path: str | Path) -> RuntimeScene:
    resolved_path = Path(xml_path).resolve()
    spec = load_scene_spec(resolved_path)
    return runtime_scene_from_spec(spec, resolved_path)


def save_scene_spec(spec: mujoco.MjSpec, xml_path: str | Path) -> Path:
    output_path = Path(xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.to_file(str(output_path))
    return output_path


def save_runtime_scene(
    scene: RuntimeScene,
    xml_path: str | Path,
    metadata_path: str | Path | None = None,
) -> Path:
    saved_xml_path = save_scene_spec(scene.spec, xml_path)
    if metadata_path is not None:
        output_metadata_path = Path(metadata_path)
        output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with output_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(scene.metadata, f, indent=4)
    return saved_xml_path

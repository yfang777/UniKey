from __future__ import annotations

import yaml

from robot.model import resolve_robot_paths

from .task_scene_generators import build_task_scene_generator
from .utils.physics_config import load_task_physics_config, normalize_physics_config


class SceneGenerator:
    def __init__(
        self,
        task_name,
        range_config_path,
        object_dir,
        physics_cfg=None,
        tasks_config_path="configurations/tasks.yaml",
        **kwargs,
    ):
        self.task_name = task_name
        self.scene_task_name = task_name
        self.cfg = self.load_yaml(range_config_path)
        tasks_config = self.load_yaml(tasks_config_path)
        task_cfg = tasks_config["tasks"][task_name]
        cameras = {name: tasks_config["camera_params"][name] for name in task_cfg["camera_names"]}
        self.robot_variant = task_cfg["robot_variant"]
        self.urdf_path, self.gripper_xml_path = resolve_robot_paths(self.robot_variant)
        if self.robot_variant == "wo_wrist" and any(p["parent_frame"] != "world" for p in cameras.values()):
            raise ValueError("wo_wrist cannot select a robot-mounted camera")
        self.object_dir = object_dir
        self.physics_cfg = (
            normalize_physics_config(physics_cfg)
            if physics_cfg is not None
            else load_task_physics_config(task_name, tasks_config_path)
        )
        self._generator = build_task_scene_generator(
            self.scene_task_name,
            self.cfg,
            self.gripper_xml_path,
            self.object_dir,
            physics_cfg=self.physics_cfg,
            cameras=cameras,
            **kwargs,
        )
        self.bounds = self._generator.bounds

    def load_yaml(self, path):
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def generate_scene(self, sample_params, output_cls_dir, sample_idx):
        return self._generator.generate_scene(sample_params, output_cls_dir, sample_idx)

    def generate_runtime_scene(self, sample_params, output_cls_dir, sample_idx):
        return self._generator.generate_runtime_scene(sample_params, output_cls_dir, sample_idx)

    def sample_params(self, n_samples, seed):
        return self._generator.sample_params(n_samples, seed)

    def generate_samples(self, base_dir, n_samples, seed):
        self._generator.generate_samples(base_dir, n_samples, seed)

"""Build the scene sampler for one of the released tasks."""

from pathlib import Path

import yaml

from keypoint_program_solver.utils import resolve_robot_paths
from .task_scene_generators import TASK_SCENE_GENERATORS


def make_generator(task, config):
    task_cfg = config["tasks"][task]
    cameras = {name: config["camera_params"][name] for name in task_cfg["camera_names"]}
    _, robot_xml = resolve_robot_paths(task_cfg["robot_variant"])
    ranges = Path("configurations/initializations_range") / task / f"{task}.yaml"
    return TASK_SCENE_GENERATORS[task](
        yaml.safe_load(ranges.read_text()), robot_xml, Path("assets/objects") / task,
        physics_cfg=task_cfg["physics"], cameras=cameras,
    )

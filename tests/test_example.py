import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from robot.model import scene_model_text
from scene_generator.generator import SceneGenerator
from rekep_solver.episode_constraints import bind_episode_constraints
from scripts.utils.evaluate import check_geometric_constraints, load_module


class ExampleTests(unittest.TestCase):
    def test_scene_loads_with_both_cameras_and_target_plate(self):
        generator = SceneGenerator(
            "arrange_cube_by_color",
            "configurations/initializations_range/arrange_cube_by_color/arrange_cube_by_color.yaml",
            "assets/objects/arrange_cube_by_color",
        )
        with tempfile.TemporaryDirectory() as directory:
            scenes = Path(directory) / "scenes"
            generator.generate_samples(scenes, 1, 0)
            model = mujoco.MjModel.from_xml_path(str(scenes / "sample_0/tmp_scene.xml"))
            self.assertEqual(scene_model_text(model, "observation_cameras"), "front wrist")
            self.assertEqual(scene_model_text(model, "robot_variant"), "w_wrist")
            self.assertEqual(model.ncam, 2)
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_plate"), 0)
            marker = scenes / "keep.txt"
            marker.write_text("existing data")
            with self.assertRaises(FileExistsError):
                generator.generate_samples(scenes, 1, 0)
            self.assertEqual(marker.read_text(), "existing data")

    def test_placement_judge_uses_selected_plate(self):
        module = load_module(
            "assets/solution/arrange_cube_by_color/constraints_judge.py", "test_judge",
        )
        for target in (3, 4, 5):
            with self.subTest(target=target):
                keypoints = np.zeros((8, 3))
                keypoints[3:6] = [[0.32, -0.1, 0.008], [0.22, 0.05, 0.008], [0.22, -0.25, 0.008]]
                keypoints[1] = keypoints[target] + [0, 0, 0.01425]
                keypoints[2] = keypoints[1] + [0, -0.01225, 0]
                judge = bind_episode_constraints(module, np.array([target] + [0] * 9))
                self.assertEqual(check_geometric_constraints(judge, keypoints, 5, -0.005), (True, True))
                misplaced = keypoints.copy()
                misplaced[1:3, 0] += 0.1
                self.assertEqual(check_geometric_constraints(judge, misplaced, 5, -0.005), (False, True))


if __name__ == "__main__":
    unittest.main()

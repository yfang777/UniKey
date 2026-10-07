"""LeRobot features for the cube-color example."""

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def dataset_features(camera_names, camera_params):
    pose_names = ["ee_x", "ee_y", "ee_z"] + [f"ee_R_{i}" for i in range(1, 7)]
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (10,), "names": pose_names + ["gripper_norm"],
        },
        "action": {
            "dtype": "float32", "shape": (10,), "names": pose_names + ["gripper"],
        },
        "observation.keypoints": {
            "dtype": "float32", "shape": (24,),
            "names": [f"keypoint_{i}_{axis}" for i in range(8) for axis in "xyz"],
        },
        "observation.constraint_context": {
            "dtype": "float32", "shape": (10,),
            "names": ["target_plate_keypoint"] + [
                f"initial_{name}_{axis}"
                for name in ("tcp", "left_finger", "right_finger") for axis in "xyz"
            ],
        },
        "success": {"dtype": "int64", "shape": (1,), "names": ["success"]},
        "elapsed_time": {"dtype": "float32", "shape": (1,), "names": ["elapsed_time"]},
        "last_gripper_status": {"dtype": "float32", "shape": (1,), "names": ["last_gripper_status"]},
    }
    for name in camera_names:
        width, height = camera_params[name]["resolution"]
        features[f"observation.images.{name}"] = {
            "dtype": "video", "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def create_dataset(root: Path, task_cfg: dict, camera_params: dict):
    return LeRobotDataset.create(
        repo_id="unikey/arrange_cube_by_color", root=root, fps=50,
        features=dataset_features(task_cfg["camera_names"], camera_params),
        robot_type=f"xarm7_{task_cfg['robot_variant']}", use_videos=True,
        image_writer_processes=0, image_writer_threads=4,
        vcodec="h264", video_backend="pyav",
    )

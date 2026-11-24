"""LeRobot features for the demonstration tasks."""

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def dataset_features(task, camera_names, camera_params):
    keypoints = {"hang_mug": 13, "arrange_table": 15, "flip_then_arrange_shoe": 18}[task]
    pose_names = ["ee_x", "ee_y", "ee_z"] + [f"ee_R_{i}" for i in range(1, 7)]
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (10,), "names": pose_names + ["gripper_norm"],
        },
        "action": {
            "dtype": "float32", "shape": (10,), "names": pose_names + ["gripper"],
        },
        "observation.keypoints": {
            "dtype": "float32", "shape": (3 * keypoints,),
            "names": [f"keypoint_{i}_{axis}" for i in range(keypoints) for axis in "xyz"],
        },
        "success": {"dtype": "int64", "shape": (1,), "names": ["success"]},
        "elapsed_time": {"dtype": "float32", "shape": (1,), "names": ["elapsed_time"]},
        "last_gripper_status": {"dtype": "float32", "shape": (1,), "names": ["last_gripper_status"]},
    }
    if task == "flip_then_arrange_shoe":
        for shoe in ("shoe_right", "shoe_left"):
            features[f"object.{shoe}.pose"] = {
                "dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
            }
    for name in camera_names:
        width, height = camera_params[name]["resolution"]
        features[f"observation.images.{name}"] = {
            "dtype": "video", "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def create_dataset(root: Path, task: str, task_cfg: dict, camera_params: dict):
    return LeRobotDataset.create(
        repo_id=f"unikey/{task}", root=root, fps=50,
        features=dataset_features(task, task_cfg["camera_names"], camera_params),
        robot_type=f"xarm7_{task_cfg['robot_variant']}", use_videos=True,
        streaming_encoding=True, encoder_queue_maxsize=4, encoder_threads=1,
        vcodec="h264", video_backend="pyav",
    )

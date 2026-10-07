"""Execute scene samples, judge placement, and write a local LeRobot dataset."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rekep_solver import StableExecutor
from scripts.utils.evaluate import evaluate_episode, load_module
from scripts.utils.features import create_dataset, dataset_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, nargs="+", required=True)
    parser.add_argument("--root_path", type=Path, required=True)
    args = parser.parse_args()
    if args.root_path.exists():
        parser.error("--root_path must be a new directory")
    for scene in args.scene_list:
        if not scene.is_file():
            parser.error(f"Scene does not exist: {scene}")

    cfg = yaml.safe_load(Path("configurations/tasks.yaml").read_text())
    task_cfg = cfg["tasks"]["arrange_cube_by_color"]
    features = dataset_features(task_cfg["camera_names"], cfg["camera_params"])
    judge = load_module(task_cfg["constraint_judge_path"], "constraints_judge")
    dataset = create_dataset(args.root_path, task_cfg, cfg["camera_params"])
    results = []
    state_records = []
    try:
        for scene in args.scene_list:
            executor = StableExecutor(max_ee_linear_vel=task_cfg["max_ee_linear_vel"])
            started = time.monotonic()
            try:
                completed, _ = executor.perform_task_fixed(
                    scene_xml_path=scene,
                    module_filename=task_cfg["stable_solution_path"],
                    dummy_simulate_t=task_cfg["dummy_simulate_t"],
                )
                if executor.env.cams != task_cfg["camera_names"]:
                    raise ValueError("Scene cameras do not match the task configuration")
                if executor.env.robot_variant != task_cfg["robot_variant"]:
                    raise ValueError("Scene robot variant does not match the task configuration")
                frames = executor.lerobot_features
                passed, last_stage, last_frame = evaluate_episode(
                    pd.DataFrame(frames), judge, judge.grasp_stage,
                    task_cfg["rekep_tolerance"],
                )
                passed = bool(completed and passed)
                elapsed = time.monotonic() - started
                episode_index = dataset.num_episodes if passed else None
                results.append({
                    "scene": str(scene), "success": passed,
                    "episode_index": episode_index, "frames": len(frames),
                    "last_success_stage": last_stage, "last_success_frame": last_frame,
                    "elapsed_seconds": elapsed,
                })
                if passed:
                    for index, frame in enumerate(frames):
                        values = {**frame, "success": [1], "elapsed_time": [elapsed]}
                        record = {
                            key: (values[key] if spec["dtype"] == "video"
                                  else np.asarray(values[key], dtype=spec["dtype"]).reshape(spec["shape"]))
                            for key, spec in features.items()
                        }
                        dataset.add_frame({**record, "task": task_cfg["task_description"]})
                        state_records.append({
                            "episode_index": episode_index, "frame_index": index,
                            "state_buf": np.asarray(frame["state_buf"], dtype=np.float64),
                            "last_gripper_status": np.asarray(frame["last_gripper_status"], dtype=np.float64),
                        })
                    dataset.save_episode()
                print(json.dumps(results[-1]), flush=True)
            finally:
                executor.close()
    finally:
        dataset.finalize()
        (args.root_path / "generation.json").write_text(json.dumps(results, indent=2) + "\n")
        if state_records:
            state_dir = args.root_path / "meta/state_buffers/chunk-000"
            state_dir.mkdir(parents=True)
            pd.DataFrame(state_records).to_parquet(state_dir / "file-000.parquet", index=False)

    if not any(result["success"] for result in results):
        raise SystemExit("No episodes passed the placement constraints; see generation.json")


if __name__ == "__main__":
    main()

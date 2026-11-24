"""Check the exported episodes, replay states, judgments and decoded camera streams."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import yaml

from keypoint_program_solver import constraint_helpers
from policy.evaluate import bind_helpers, evaluate_episode, load_module


def validate(root: Path):
    info = json.loads((root / "meta/info.json").read_text())
    report = json.loads((root / "generation.json").read_text())
    cfg = yaml.safe_load(Path("configurations/tasks.yaml").read_text())["tasks"][report["task"]]
    judge = load_module(cfg["constraint_judge_path"], "constraints_judge")
    bind_helpers(judge, constraint_helpers)
    frames, states, episodes = (
        pd.concat([pd.read_parquet(p) for p in sorted(directory.rglob("*.parquet"))], ignore_index=True)
        for directory in (root / "data", root / "meta/state_buffers", root / "meta/episodes")
    )
    if not (0 < len(episodes) == report["episodes"] == info["total_episodes"]):
        raise ValueError("Episode count does not match metadata")
    if len(frames) != info["total_frames"] or not np.array_equal(frames["index"], np.arange(len(frames))):
        raise ValueError("Invalid global frame indices")
    keys = ["episode_index", "frame_index"]
    if not frames[keys].equals(states[keys]):
        raise ValueError("Replay states do not align with trajectory frames")
    if not np.array_equal(frames["last_gripper_status"].to_numpy(), np.stack(states["last_gripper_status"]).ravel()):
        raise ValueError("Replay gripper state mismatch")
    for name, spec in info["features"].items():
        if name in frames and name not in ("timestamp", "index", "episode_index", "frame_index", "task_index"):
            values = np.stack(frames[name])
            if spec["shape"] == [1] and values.ndim == 1:
                values = values[:, None]
            if values.shape[1:] != tuple(spec["shape"]) or not np.isfinite(values).all():
                raise ValueError(f"Invalid values in {name}")
    if not np.isfinite(np.stack(states["state_buf"])).all() or not (np.stack(frames["success"]) == 1).all():
        raise ValueError("Invalid replay state or unsuccessful episode")
    passed = [attempt for attempt in report["attempts"] if attempt["success"]]
    if [a["episode_index"] for a in passed] != list(range(len(episodes))):
        raise ValueError("Missing or duplicate successful episode")
    video_counts = {f"observation.images.{cam}": 0 for cam in cfg["camera_names"]}
    for episode_id, attempt in enumerate(passed):
        meta = episodes.iloc[episode_id]
        batch = frames.iloc[int(meta["dataset_from_index"]):int(meta["dataset_to_index"])]
        expected = np.arange(len(batch))
        if (int(meta["episode_index"]) != episode_id or not (batch["episode_index"] == episode_id).all()
                or len(batch) != meta["length"] or len(batch) != attempt["frames"]
                or not np.array_equal(batch["frame_index"], expected)
                or not np.allclose(batch["timestamp"], expected / info["fps"], atol=1e-5)):
            raise ValueError(f"Invalid trajectory metadata for episode {episode_id}")
        if getattr(judge, "uses_episode_context", False):
            batch = batch.assign(**{"observation.constraint_context": list(states["constraint_context"].iloc[batch.index])})
        judged, stage, frame = evaluate_episode(batch, judge, judge.grasp_stage, cfg["rekep_tolerance"])
        if (judged, stage, frame) != (True, attempt["last_success_stage"], attempt["last_success_frame"]):
            raise ValueError(f"Saved trajectory failed its recorded judgment for episode {episode_id}")
        for cam in video_counts:
            prefix = f"videos/{cam}"
            path = root / info["video_path"].format(video_key=cam,
                chunk_index=int(meta[prefix + "/chunk_index"]), file_index=int(meta[prefix + "/file_index"]))
            height, width, _ = info["features"][cam]["shape"]
            # The generator writes one video file per camera per episode.
            with av.open(str(path)) as container:
                if container.streams.video[0].average_rate != info["fps"]:
                    raise ValueError(f"Wrong video rate in {path}")
                count = 0
                for image in container.decode(video=0):
                    if (image.height, image.width) != (height, width):
                        raise ValueError(f"Wrong video size in {path}")
                    count += 1
            if (count != len(batch) or not np.isclose(meta[prefix + "/from_timestamp"], 0)
                    or not np.isclose(meta[prefix + "/to_timestamp"], count / info["fps"])):
                raise ValueError(f"Wrong video interval in {path}")
            video_counts[cam] += count
    return dict(episodes=len(episodes), frames=len(frames), video_frames=video_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(validate(parser.parse_args().root), indent=2))

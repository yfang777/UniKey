"""Validate trajectory, metadata, state-buffer, and video exports."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd


def validate(root: Path):
    info = json.loads((root / "meta/info.json").read_text())
    frames = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    states = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/state_buffers").rglob("*.parquet"))])
    if len(frames) == 0 or len(frames) != info["total_frames"]:
        raise ValueError("Frame count does not match metadata")
    if frames["episode_index"].nunique() != info["total_episodes"]:
        raise ValueError("Episode count does not match metadata")
    if not frames[["episode_index", "frame_index"]].reset_index(drop=True).equals(
        states[["episode_index", "frame_index"]].reset_index(drop=True)
    ):
        raise ValueError("Simulation states do not align with trajectory frames")
    for name in ("observation.state", "action", "observation.keypoints", "observation.constraint_context"):
        values = np.stack(frames[name])
        if values.shape[1:] != tuple(info["features"][name]["shape"]) or not np.isfinite(values).all():
            raise ValueError(f"Invalid shape or nonfinite values in {name}")
    if not (np.stack(frames["success"]) == 1).all():
        raise ValueError("Dataset contains an unsuccessful episode")
    for _, episode in frames.groupby("episode_index", sort=False):
        expected = np.arange(len(episode))
        if not np.array_equal(episode["frame_index"], expected):
            raise ValueError("Frame indices are not consecutive")
        if not np.allclose(episode["timestamp"], expected / info["fps"], atol=1e-5):
            raise ValueError("Frame timestamps do not match frame rate")
    video_counts = {}
    for name in ("observation.images.front", "observation.images.wrist"):
        height, width, _ = info["features"][name]["shape"]
        count = 0
        for path in sorted((root / "videos" / name).rglob("*.mp4")):
            with av.open(str(path)) as video:
                for frame in video.decode(video=0):
                    if (frame.height, frame.width) != (height, width):
                        raise ValueError(f"Wrong resolution in {path}")
                    count += 1
        if count != len(frames):
            raise ValueError(f"Video frame count mismatch for {name}: {count} vs {len(frames)}")
        video_counts[name] = count
    return {"episodes": info["total_episodes"], "frames": len(frames), "video_frames": video_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2))


if __name__ == "__main__":
    main()

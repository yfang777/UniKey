#!/usr/bin/env python3
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import load_stats, write_stats


DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


@dataclass
class SourceDataset:
    root: Path
    info: dict
    episodes: pd.DataFrame
    tasks: pd.DataFrame
    stats: dict
    state_buffers: pd.DataFrame | None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_source_dataset(src_dir: Path) -> SourceDataset | None:
    info_path = src_dir / "meta" / "info.json"
    tasks_path = src_dir / "meta" / "tasks.parquet"
    episode_paths = sorted((src_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    state_buffer_paths = sorted(
        (src_dir / "meta" / "state_buffers").glob("chunk-*/file-*.parquet")
    )

    if not info_path.exists() or not tasks_path.exists() or not episode_paths:
        print(f"[skip empty chunk] {src_dir}")
        return None

    episodes = pd.concat(
        [pd.read_parquet(path) for path in episode_paths],
        ignore_index=True,
    )
    if episodes.empty:
        print(f"[skip empty chunk] {src_dir}")
        return None

    return SourceDataset(
        root=src_dir,
        info=load_json(info_path),
        episodes=episodes.sort_values("episode_index").reset_index(drop=True),
        tasks=pd.read_parquet(tasks_path),
        stats=load_stats(src_dir),
        state_buffers=(
            pd.concat(
                [pd.read_parquet(path) for path in state_buffer_paths],
                ignore_index=True,
            )
            if state_buffer_paths
            else None
        ),
    )


def combined_tasks(sources: list[SourceDataset]) -> pd.DataFrame:
    task_names = []
    for source in sources:
        ordered_tasks = source.tasks.sort_values("task_index")
        for task in ordered_tasks.index:
            if task not in task_names:
                task_names.append(task)

    return pd.DataFrame(
        {"task_index": range(len(task_names))},
        index=pd.Index(task_names, name="task"),
    )


def task_index_map(source: SourceDataset, tasks: pd.DataFrame) -> dict[int, int]:
    mapping = {}
    for task, row in source.tasks.iterrows():
        mapping[int(row["task_index"])] = int(tasks.loc[task]["task_index"])
    return mapping


def copy_file(src: Path, dst: Path, copy_mode: str) -> None:
    ensure_parent(dst)
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "move":
        shutil.move(src, dst)
    elif copy_mode == "link":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    elif copy_mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)


def source_data_path(source: SourceDataset, episode: pd.Series) -> Path:
    return source.root / source.info["data_path"].format(
        chunk_index=int(episode["data/chunk_index"]),
        file_index=int(episode["data/file_index"]),
    )


def source_video_path(source: SourceDataset, episode: pd.Series, video_key: str) -> Path:
    return source.root / source.info["video_path"].format(
        video_key=video_key,
        chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
        file_index=int(episode[f"videos/{video_key}/file_index"]),
    )


def shift_stat(value, offset: int):
    return np.asarray(value) + offset


def constant_stat(value, constant: int):
    return np.full_like(np.asarray(value), constant)


def zero_stat(value):
    return np.zeros_like(np.asarray(value))


def serialize_parquet_value(value):
    if isinstance(value, np.ndarray):
        return serialize_parquet_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list | tuple):
        return [serialize_parquet_value(item) for item in value]
    return value


def serialize_episode_row(episode: pd.Series) -> dict:
    return {
        key: serialize_parquet_value(value)
        for key, value in episode.items()
    }


def update_episode_stats(
    episode: pd.Series,
    new_episode_index: int,
    index_offset: int,
    new_task_index: int | None,
) -> pd.Series:
    for column in episode.index:
        if column.startswith("stats/index/"):
            if column.endswith("/std") or column.endswith("/count"):
                continue
            episode[column] = shift_stat(episode[column], index_offset)
        elif column.startswith("stats/episode_index/"):
            if column.endswith("/count"):
                continue
            episode[column] = zero_stat(episode[column]) if column.endswith("/std") else constant_stat(
                episode[column], new_episode_index
            )
        elif new_task_index is not None and column.startswith("stats/task_index/"):
            if column.endswith("/count"):
                continue
            episode[column] = zero_stat(episode[column]) if column.endswith("/std") else constant_stat(
                episode[column], new_task_index
            )
    return episode


def update_generation_info(src_dirs: list[Path], episode_id_map: dict[tuple[Path, int], int], dst_root: Path) -> None:
    all_gen_info = []

    for src_dir in src_dirs:
        gen_info_path = src_dir / "generation.info"
        if not gen_info_path.exists():
            continue

        with gen_info_path.open("r") as f:
            info_list = json.load(f)

        for item in info_list:
            if item.get("success", False) and item.get("episode") != "NA":
                old_id = int(item["episode"])
                key = (src_dir, old_id)
                if key in episode_id_map:
                    item["episode"] = episode_id_map[key]
                else:
                    item["episode"] = "NA"
                    item["success"] = False
            all_gen_info.append(item)

    if all_gen_info:
        with (dst_root / "generation.info").open("w") as f:
            json.dump(all_gen_info, f, indent=4)


def consolidate(
    src_dirs: list[Path],
    dst_root: Path,
    camera_keys: list[str],
    chunk_size: int,
    copy_mode: str = "copy",
    skip_videos: bool = False,
) -> None:
    assert copy_mode in {"copy", "move", "link", "symlink"}

    sources = [source for source in (load_source_dataset(src_dir) for src_dir in src_dirs) if source is not None]
    if not sources:
        print("No valid episodes found in any source directory.")
        return

    tasks = combined_tasks(sources)
    task_maps = {source.root: task_index_map(source, tasks) for source in sources}

    base_info = dict(sources[0].info)
    # Older recovery datasets do not record this optional annotation.
    include_success = all("success" in source.info["features"] for source in sources)
    if not include_success:
        base_info["features"] = {
            key: value for key, value in base_info["features"].items() if key != "success"
        }
    base_info["data_path"] = DEFAULT_DATA_PATH
    base_info["video_path"] = None if skip_videos else DEFAULT_VIDEO_PATH
    base_info["chunks_size"] = chunk_size
    base_info.pop("data_root", None)
    base_info.pop("total_videos", None)
    base_info.pop("total_chunks", None)

    data_cache: dict[Path, pd.DataFrame] = {}
    video_file_map: dict[tuple[Path, str, int, int], tuple[int, int]] = {}
    video_file_count = {video_key: 0 for video_key in camera_keys}
    out_dataframes = []
    out_episodes = []
    out_state_buffers = []
    episode_id_map: dict[tuple[Path, int], int] = {}
    total_frames = 0
    new_episode_index = 0

    for source in sources:
        source_task_map = task_maps[source.root]
        for _, episode in source.episodes.iterrows():
            old_episode_index = int(episode["episode_index"])
            data_path = source_data_path(source, episode)
            if data_path not in data_cache:
                data_cache[data_path] = pd.read_parquet(data_path)

            data = data_cache[data_path]
            episode_data = data[data["episode_index"] == old_episode_index].copy()
            if not include_success and "success" in episode_data.columns:
                episode_data = episode_data.drop(columns="success")
            if episode_data.empty:
                raise ValueError(f"No frames for episode {old_episode_index} in {data_path}")

            episode_length = len(episode_data)
            old_dataset_from_index = int(episode["dataset_from_index"])
            index_offset = total_frames - old_dataset_from_index

            episode_data["episode_index"] = np.int64(new_episode_index)
            episode_data["frame_index"] = np.arange(episode_length, dtype=np.int64)
            episode_data["timestamp"] = episode_data["frame_index"].astype(np.float32) / float(base_info["fps"])
            episode_data["index"] = np.arange(total_frames, total_frames + episode_length, dtype=np.int64)
            episode_data["task_index"] = episode_data["task_index"].map(source_task_map).astype(np.int64)
            out_dataframes.append(episode_data)

            if source.state_buffers is not None:
                state_data = source.state_buffers[
                    source.state_buffers["episode_index"] == old_episode_index
                ].copy()
                if len(state_data) != episode_length:
                    raise ValueError(
                        f"Expected {episode_length} state-buffer rows for episode "
                        f"{old_episode_index} in {source.root}, got {len(state_data)}"
                    )
                state_data = state_data.sort_values("frame_index").reset_index(drop=True)
                state_data["episode_index"] = np.int64(new_episode_index)
                state_data["frame_index"] = np.arange(episode_length, dtype=np.int64)
                if "index" in state_data.columns:
                    state_data["index"] = np.arange(
                        total_frames,
                        total_frames + episode_length,
                        dtype=np.int64,
                    )
                out_state_buffers.append(state_data)

            out_episode = episode.copy()
            if not include_success:
                out_episode = out_episode.drop(
                    labels=[key for key in out_episode.index if key.startswith("stats/success/")]
                )
            out_episode["episode_index"] = np.int64(new_episode_index)
            out_episode["length"] = np.int64(episode_length)
            out_episode["data/chunk_index"] = np.int64(0)
            out_episode["data/file_index"] = np.int64(0)
            out_episode["dataset_from_index"] = np.int64(total_frames)
            out_episode["dataset_to_index"] = np.int64(total_frames + episode_length)
            out_episode["meta/episodes/chunk_index"] = np.int64(0)
            out_episode["meta/episodes/file_index"] = np.int64(0)

            episode_tasks = list(out_episode["tasks"])
            new_task_index = int(tasks.loc[episode_tasks[0]]["task_index"]) if len(episode_tasks) == 1 else None
            out_episode = update_episode_stats(
                out_episode,
                new_episode_index=new_episode_index,
                index_offset=index_offset,
                new_task_index=new_task_index,
            )

            if not skip_videos:
                for video_key in camera_keys:
                    old_video_chunk = int(episode[f"videos/{video_key}/chunk_index"])
                    old_video_file = int(episode[f"videos/{video_key}/file_index"])
                    video_key_tuple = (source.root, video_key, old_video_chunk, old_video_file)
                    src_video = source_video_path(source, episode, video_key)

                    if video_key_tuple not in video_file_map:
                        new_video_file_id = video_file_count[video_key]
                        video_file_count[video_key] += 1
                        new_video_chunk = new_video_file_id // chunk_size
                        new_video_file = new_video_file_id % chunk_size
                        dst_video = dst_root / DEFAULT_VIDEO_PATH.format(
                            video_key=video_key,
                            chunk_index=new_video_chunk,
                            file_index=new_video_file,
                        )
                        copy_file(src_video, dst_video, copy_mode)
                        video_file_map[video_key_tuple] = (new_video_chunk, new_video_file)

                    new_video_chunk, new_video_file = video_file_map[video_key_tuple]
                    out_episode[f"videos/{video_key}/chunk_index"] = np.int64(new_video_chunk)
                    out_episode[f"videos/{video_key}/file_index"] = np.int64(new_video_file)

            out_episodes.append(serialize_episode_row(out_episode))
            episode_id_map[(source.root, old_episode_index)] = new_episode_index
            total_frames += episode_length
            new_episode_index += 1

    combined_data = pd.concat(out_dataframes, ignore_index=True)
    data_path = dst_root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    ensure_parent(data_path)
    combined_data.to_parquet(data_path, index=False)

    episodes_path = dst_root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    ensure_parent(episodes_path)
    pd.DataFrame(out_episodes).to_parquet(episodes_path, index=False)

    if out_state_buffers:
        state_buffers_path = (
            dst_root / "meta" / "state_buffers" / "chunk-000" / "file-000.parquet"
        )
        ensure_parent(state_buffers_path)
        pd.concat(out_state_buffers, ignore_index=True).to_parquet(
            state_buffers_path,
            index=False,
        )

    tasks_path = dst_root / "meta" / "tasks.parquet"
    ensure_parent(tasks_path)
    tasks.to_parquet(tasks_path)

    base_info["total_episodes"] = new_episode_index
    base_info["total_frames"] = total_frames
    base_info["total_tasks"] = len(tasks)
    base_info["splits"] = {"train": f"0:{new_episode_index}"}
    save_json(dst_root / "meta" / "info.json", base_info)

    stats_list = [source.stats for source in sources if source.stats is not None]
    if not include_success:
        stats_list = [{key: value for key, value in stats.items() if key != "success"}
                      for stats in stats_list]
    if stats_list:
        write_stats(aggregate_stats(stats_list), dst_root)

    update_generation_info([source.root for source in sources], episode_id_map, dst_root)

"""Generate and assemble the individual scene datasets used by dense sampling."""

from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


def scene_paths(task, directory, start, end):
    from policy.scene_generator.generator import make_generator

    directory = Path(directory)
    generator = None
    paths = []
    for index in range(start, end):
        scene = directory / f"sample_{index}" / "tmp_scene.xml"
        if not scene.is_file():
            if generator is None:
                config = yaml.safe_load(Path("configurations/tasks.yaml").read_text())
                generator = make_generator(task, config)
            seed = index if directory.name == "train" else 2**32 + index
            params = generator.sample_params(1, seed)[0]
            generator.generate_scene(params, directory, index)
        paths.append(str(scene.resolve()))
    return paths


def _generate(job):
    task, scene, root, egl_device = job
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    from policy.data_generation.data_generate import collect_episode

    dest = root / Path(scene).parent.name
    if (dest / "generation.info").is_file():
        return str(dest)
    with TemporaryDirectory(dir=root) as tmp:
        shard, records = collect_episode(task, Path(tmp), 0, 0, 1, 1, scene_path=scene)
        if shard is None:
            return None
        states = shard / "meta/state_buffers/chunk-000/file-000.parquet"
        states.parent.mkdir(parents=True)
        (shard / "states.parquet").rename(states)
        records[0].update(episode=0, xml=str(Path(scene).resolve()))
        (shard / "generation.info").write_text(json.dumps(records))
        shard.rename(dest)
        return str(dest)


def run_parallel_generation(scenes, step_name, args, workers):
    root = Path(args.tmp_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"{step_name}: generating {len(scenes)} scenes", flush=True)
    jobs = [(args.task, scene, root, args.mujoco_egl_device_id) for scene in scenes]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        return [path for path in pool.map(_generate, jobs) if path is not None]


def concatenate_datasets(src_dirs, final_dst, camera_keys):
    from .concatenate_dataset import consolidate

    dst = Path(final_dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dst.parent) as tmp:
        staging = Path(tmp) / "dataset"
        consolidate([Path(p) for p in src_dirs], staging, camera_keys, chunk_size=1000)
        staging.rename(dst)

from pathlib import Path


def resolve_robot_paths(robot_variant: str) -> tuple[Path, Path]:
    """Return the matching kinematics URDF and simulation XML."""
    if robot_variant not in {"w_wrist", "wo_wrist"}:
        raise ValueError(f"Unknown robot_variant: {robot_variant!r}")
    asset_dir = Path(__file__).resolve().parents[1] / "assets/xarm7_gripper"
    return (
        asset_dir / f"xarm7_{robot_variant}.urdf",
        asset_dir / f"xarm7_{robot_variant}.xml",
    )


def scene_model_text(model, name: str) -> str:
    """Read required text metadata embedded in a saved MuJoCo scene."""
    import mujoco

    text_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if text_id == -1:
        raise ValueError(f"Scene has no {name}; regenerate it using the task configuration.")
    start = model.text_adr[text_id]
    size = model.text_size[text_id]
    return model.text_data[start : start + size - 1].decode("utf-8")

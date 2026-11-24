"""Process settings shared by the Codegen_Datagen training drivers."""

import ctypes
import os
from pathlib import Path


def cache_environment(cache_root: Path) -> dict[str, str]:
    return {
        "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
        "WANDB_DIR": str(cache_root / "wandb"),
        "WANDB_CACHE_DIR": str(cache_root / "wandb" / "cache"),
        "WANDB_CONFIG_DIR": str(cache_root / "wandb" / "config"),
        "WANDB_DATA_DIR": str(cache_root / "wandb" / "data"),
        "HF_HOME": str(cache_root / "huggingface"),
        "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(cache_root / "huggingface" / "transformers"),
        "TORCH_HOME": os.environ.get("TORCH_HOME", str(Path.home() / ".cache/torch")),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "OPENPI_DATA_HOME": str(cache_root / "openpi"),
        "JAX_COMPILATION_CACHE_DIR": str(cache_root / "jax"),
    }


def mujoco_egl_device_id_for_cuda_device(cuda_device_id: str) -> str:
    from OpenGL import EGL
    from mujoco.egl import egl_ext

    query_device_attrib = ctypes.CFUNCTYPE(
        EGL.EGLBoolean,
        EGL.EGLDeviceEXT,
        EGL.EGLint,
        ctypes.POINTER(ctypes.c_long),
    )(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
    egl_cuda_device_attr = 0x323A
    # EGL_CUDA_DEVICE_NV uses CUDA ordinals after CUDA_VISIBLE_DEVICES remapping.
    target_cuda_device_id = int(cuda_device_id)
    for egl_device_id, egl_device in enumerate(egl_ext.eglQueryDevicesEXT()):
        queried_cuda_device_id = ctypes.c_long(-1)
        success = query_device_attrib(
            egl_device,
            egl_cuda_device_attr,
            ctypes.byref(queried_cuda_device_id),
        )
        if (
            success == EGL.EGL_TRUE
            and queried_cuda_device_id.value == target_cuda_device_id
        ):
            return str(egl_device_id)
    raise ValueError(f"No MuJoCo EGL device maps to CUDA device {cuda_device_id}")


def egl_device(device):
    if device == "cpu":
        return os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
    return mujoco_egl_device_id_for_cuda_device(device.partition(":")[2] or "0")

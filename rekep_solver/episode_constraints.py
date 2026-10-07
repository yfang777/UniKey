"""Bind scene-dependent constraints without changing the imported task module."""

from functools import partial, update_wrapper
from types import SimpleNamespace

import numpy as np


def bind_episode_constraints(module, episode_context):
    if not getattr(module, "uses_episode_context", False):
        return module
    context = np.array(episode_context, dtype=float, copy=True)
    if getattr(module, 'episode_context_kind', None) == 'shoe_tray_placement':
        if context.shape != (33,) or not np.isfinite(context).all():
            raise ValueError('Expected initial shoe frame, initial gripper, and current lift/retreat entry gripper XYZ')
    elif getattr(module, 'episode_context_kind', None) == 'shoe_insert_lift':
        if context.shape != (15,) or not np.isfinite(context).all():
            raise ValueError('Expected five initial right-shoe XYZ positions')
    else:
        expected_shape = getattr(module, "episode_context_shape", (10,))
        if context.shape != expected_shape:
            raise ValueError(f"Expected episode context shape {expected_shape}, got {context.shape}")
        if expected_shape == (10,) and context[0] not in (3, 4, 5):
            raise ValueError("Expected target plate keypoint (3/4/5) and initial TCP/left/right XYZ")
    context.setflags(write=False)
    members = dict(vars(module))
    for name, function in vars(module).items():
        if name.startswith("stage") and "_constraint" in name and callable(function):
            members[name] = update_wrapper(partial(function, context=context), function)
    members["uses_episode_context"] = False
    return SimpleNamespace(**members)

"""Online temporal action ensemble for chunked policy rollouts."""

import numpy as np


def linear_upsample_chunk(chunk: np.ndarray, policy_fps: int, env_fps: int) -> np.ndarray:
    assert env_fps % policy_fps == 0, (
        f"env_fps ({env_fps}) must be an integer multiple of policy_fps ({policy_fps})"
    )
    chunk_size, action_dim = chunk.shape
    upsample = env_fps // policy_fps
    env_chunk_size = (chunk_size - 1) * upsample + 1
    t_known = np.arange(chunk_size, dtype=np.float64) / policy_fps
    t_query = np.arange(env_chunk_size, dtype=np.float64) / env_fps
    out = np.empty((env_chunk_size, action_dim), dtype=np.float64)
    for dim in range(action_dim):
        out[:, dim] = np.interp(t_query, t_known, chunk[:, dim])
    return out


class ActionEnsembleController:
    def __init__(
        self,
        policy_chunk_size: int,
        policy_fps: int,
        env_fps: int,
        action_dim: int,
        coeff: float,
        action_timestamp_offset: float,
        initial_action: np.ndarray,
        ensemble: bool,
    ):
        assert policy_chunk_size >= 1
        assert env_fps % policy_fps == 0
        self.policy_chunk_size = policy_chunk_size
        self.policy_fps = policy_fps
        self.env_fps = env_fps
        self.upsample = env_fps // policy_fps
        offset_steps = action_timestamp_offset * env_fps
        if not np.isclose(offset_steps, round(offset_steps)):
            raise ValueError(
                "action_timestamp_offset must align with the environment control grid"
            )
        self.offset_steps = int(round(offset_steps))
        if self.offset_steps < 0:
            raise ValueError("action_timestamp_offset must be non-negative")
        self.policy_env_chunk_size = (policy_chunk_size - 1) * self.upsample + 1
        self.buffer_size = self.offset_steps + self.policy_env_chunk_size
        self.action_dim = action_dim
        self.coeff = coeff
        self.ensemble = ensemble
        self.weights = np.exp(
            -coeff * np.arange(self.policy_env_chunk_size) / self.upsample
        )
        self.reset(initial_action)

    def reset(self, initial_action: np.ndarray):
        initial_action = np.asarray(initial_action, dtype=np.float64)
        assert initial_action.shape == (self.action_dim,)
        self.initial_action = initial_action.copy()
        self.weighted_sum = np.zeros(
            (self.buffer_size, self.action_dim),
            dtype=np.float64,
        )
        self.weight_sum = np.zeros(self.buffer_size, dtype=np.float64)
        self.is_first_step = True

    def step(self, chunk: np.ndarray) -> np.ndarray:
        assert chunk.shape == (self.policy_chunk_size, self.action_dim), (
            f"expected chunk shape {(self.policy_chunk_size, self.action_dim)}, "
            f"got {chunk.shape}"
        )
        chunk_env = linear_upsample_chunk(chunk, self.policy_fps, self.env_fps)
        start = self.offset_steps
        end = start + self.policy_env_chunk_size

        if self.ensemble:
            self.weighted_sum[start:end] += self.weights[:, None] * chunk_env
            self.weight_sum[start:end] += self.weights
        else:
            self.weighted_sum[start:end] = chunk_env
            self.weight_sum[start:end] = 1.0

        if self.is_first_step:
            if self.offset_steps > 0:
                alpha = (
                    np.arange(self.offset_steps, dtype=np.float64)
                    / self.offset_steps
                )
                self.weighted_sum[:self.offset_steps] = (
                    self.initial_action[None]
                    + alpha[:, None] * (chunk_env[0] - self.initial_action)[None]
                )
                self.weight_sum[:self.offset_steps] = 1.0
            self.is_first_step = False

        assert np.all(self.weight_sum[:self.upsample] > 0)
        actions = (
            self.weighted_sum[:self.upsample]
            / self.weight_sum[:self.upsample, None]
        ).astype(np.float32)

        self.weighted_sum[:-self.upsample] = self.weighted_sum[self.upsample:]
        self.weighted_sum[-self.upsample:] = 0.0
        self.weight_sum[:-self.upsample] = self.weight_sum[self.upsample:]
        self.weight_sum[-self.upsample:] = 0.0
        return actions

"""Rollout buffer: `[T, E, N, ...]` layout (§15.4).

Stores everything collected during a rollout, in the un-flattened
time/env/agent shape GAE needs (§16.8 item 2). `state_grid`/`state_vec`
are only populated when the algorithm's critic uses global state (MAPPO,
Phase 5) — for IPPO (Phase 4) they stay `None` and the critic reuses the
actor's own observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from marl.envs.core.observations import N_OBS_CHANNELS, N_VEC_FEATURES
from marl.envs.core.types import N_ACTIONS


@dataclass
class RolloutBuffer:
    t_len: int
    n_envs: int
    n_agents: int
    grid_size: int
    global_channels: int | None = None  # set only when storing MAPPO global state
    global_vec_dim: int = 4
    device: str = "cpu"

    state_grid: np.ndarray | None = None
    state_vec: np.ndarray | None = None

    def __post_init__(self) -> None:
        t, e, n, k = self.t_len, self.n_envs, self.n_agents, self.grid_size

        self.obs_grid = np.zeros((t, e, n, N_OBS_CHANNELS, k, k), dtype=np.uint8)
        self.obs_vec = np.zeros((t, e, n, N_VEC_FEATURES), dtype=np.float32)
        self.mask = np.zeros((t, e, n, N_ACTIONS), dtype=np.int8)
        self.action = np.zeros((t, e, n), dtype=np.int64)
        self.logp_old = np.zeros((t, e, n), dtype=np.float32)
        self.value_old = np.zeros((t, e, n), dtype=np.float32)  # normalized space if ValueNorm on
        self.reward = np.zeros((t, e, n), dtype=np.float32)
        self.terminated = np.zeros((t, e), dtype=bool)
        self.truncated = np.zeros((t, e), dtype=bool)
        self.active = np.ones((t, e, n), dtype=bool)

        if self.global_channels is not None:
            self.state_grid = np.zeros((t, e, self.global_channels, k, k), dtype=np.float32)
            # NOTE: global state grids are only same-shape within one
            # vector env at a fixed map size (§7.3); k here is a
            # placeholder for H,W set by the caller at construction when
            # map size is fixed per training run.
            self.state_vec = np.zeros((t, e, self.global_vec_dim), dtype=np.float32)
        else:
            self.state_grid = None
            self.state_vec = None

        self._write_idx = 0

    def add(
        self,
        obs_grid: np.ndarray,
        obs_vec: np.ndarray,
        mask: np.ndarray,
        action: np.ndarray,
        logp_old: np.ndarray,
        value_old: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        state_grid: np.ndarray | None = None,
        state_vec: np.ndarray | None = None,
    ) -> None:
        t = self._write_idx
        if t >= self.t_len:
            raise IndexError(f"RolloutBuffer is full ({self.t_len} steps already written)")

        self.obs_grid[t] = obs_grid
        self.obs_vec[t] = obs_vec
        self.mask[t] = mask
        self.action[t] = action
        self.logp_old[t] = logp_old
        self.value_old[t] = value_old
        self.reward[t] = reward
        self.terminated[t] = terminated
        self.truncated[t] = truncated

        if self.state_grid is not None and self.state_vec is not None:
            if state_grid is None or state_vec is None:
                raise ValueError("this buffer stores global state; state_grid/state_vec are required")
            self.state_grid[t] = state_grid
            self.state_vec[t] = state_vec

        self._write_idx += 1

    @property
    def is_full(self) -> bool:
        return self._write_idx == self.t_len

    @property
    def done(self) -> np.ndarray:
        """[T, E] bool — terminated OR truncated."""
        return self.terminated | self.truncated

    def to_torch(self, device: str | None = None) -> RolloutBufferTensors:
        dev = device or self.device
        return RolloutBufferTensors(
            obs_grid=torch.as_tensor(self.obs_grid, device=dev).float(),  # uint8 -> float32
            obs_vec=torch.as_tensor(self.obs_vec, device=dev),
            mask=torch.as_tensor(self.mask, device=dev).bool(),
            action=torch.as_tensor(self.action, device=dev),
            logp_old=torch.as_tensor(self.logp_old, device=dev),
            value_old=torch.as_tensor(self.value_old, device=dev),
            reward=torch.as_tensor(self.reward, device=dev),
            terminated=torch.as_tensor(self.terminated, device=dev),
            truncated=torch.as_tensor(self.truncated, device=dev),
            active=torch.as_tensor(self.active, device=dev),
            state_grid=torch.as_tensor(self.state_grid, device=dev) if self.state_grid is not None else None,
            state_vec=torch.as_tensor(self.state_vec, device=dev) if self.state_vec is not None else None,
        )


@dataclass
class RolloutBufferTensors:
    """Torch view of a `RolloutBuffer`, all still `[T, E, N, ...]`."""

    obs_grid: torch.Tensor
    obs_vec: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    logp_old: torch.Tensor
    value_old: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    active: torch.Tensor
    state_grid: torch.Tensor | None = None
    state_vec: torch.Tensor | None = None

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated


@dataclass
class FlatBatch:
    """Flattened `[B, ...]` samples (B = T*E*N), post-GAE, for minibatching."""

    obs_grid: torch.Tensor
    obs_vec: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    logp_old: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    active: torch.Tensor
    state_grid: torch.Tensor | None = None
    state_vec: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.obs_grid.shape[0]


def flatten_batch(
    buf: RolloutBufferTensors,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> FlatBatch:
    """Flatten [T,E,N,...] -> [T*E*N,...] for minibatch sampling (§16.5)."""

    def flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *x.shape[3:]) if x.dim() > 3 else x.reshape(-1)

    def flat_state(x: torch.Tensor | None) -> torch.Tensor | None:
        # state_grid/state_vec are [T,E,...] (shared per env, not per
        # agent) — expand across the N axis before flattening so each
        # (t,e,n) sample gets its env's shared state.
        if x is None:
            return None
        t, e = x.shape[0], x.shape[1]
        n = buf.action.shape[2]
        expanded = x.unsqueeze(2).expand(t, e, n, *x.shape[2:])
        return expanded.reshape(-1, *x.shape[2:])

    return FlatBatch(
        obs_grid=flat(buf.obs_grid),
        obs_vec=flat(buf.obs_vec),
        mask=flat(buf.mask),
        action=flat(buf.action),
        logp_old=flat(buf.logp_old),
        advantages=flat(advantages),
        returns=flat(returns),
        active=flat(buf.active),
        state_grid=flat_state(buf.state_grid),
        state_vec=flat_state(buf.state_vec),
    )

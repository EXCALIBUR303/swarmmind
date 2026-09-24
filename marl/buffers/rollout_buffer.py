"""Rollout buffer: `[T, E, N, ...]` layout (§15.4, §16.5).

Stores everything collected during a rollout, in the un-flattened
time/env/agent shape GAE needs (§16.8 item 2).

Global state (MAPPO only, Phase 5) is stored the way §16.5 prescribes:
the 7 SHARED channels once per `(t, e)` — they're identical for every
robot — and the 2 AGENT-SPECIFIC channels (own position, own belief mask)
per `(t, e, n)` as uint8. The full 9-channel critic input is assembled
per sample only at flatten time. With `state_hw=None` (IPPO) none of this
is allocated and the critic reuses the actor's own observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from marl.envs.core.observations import N_OBS_CHANNELS, N_VEC_FEATURES
from marl.envs.core.types import N_ACTIONS

N_SHARED_STATE_CHANNELS = 7
N_AGENT_STATE_CHANNELS = 2
N_GLOBAL_VEC = 4


@dataclass
class RolloutBuffer:
    t_len: int
    n_envs: int
    n_agents: int
    grid_size: int
    state_hw: tuple[int, int] | None = None  # (H, W) of the map; set only when storing global state
    device: str = "cpu"

    def __post_init__(self) -> None:
        t, e, n, k = self.t_len, self.n_envs, self.n_agents, self.grid_size

        self.obs_grid = np.zeros((t, e, n, N_OBS_CHANNELS, k, k), dtype=np.uint8)
        self.obs_vec = np.zeros((t, e, n, N_VEC_FEATURES), dtype=np.float32)
        self.mask = np.zeros((t, e, n, N_ACTIONS), dtype=np.int8)
        self.action = np.zeros((t, e, n), dtype=np.int64)
        self.logp_old = np.zeros((t, e, n), dtype=np.float32)
        self.value_old = np.zeros((t, e, n), dtype=np.float32)  # DENORMALIZED (reward scale)
        self.reward = np.zeros((t, e, n), dtype=np.float32)
        self.terminated = np.zeros((t, e), dtype=bool)
        self.truncated = np.zeros((t, e), dtype=bool)
        self.active = np.ones((t, e, n), dtype=bool)

        self.state_grid: np.ndarray | None = None
        self.agent_state: np.ndarray | None = None
        self.state_vec: np.ndarray | None = None
        if self.state_hw is not None:
            h, w = self.state_hw
            # Shared channels include a fractional visit-count channel, so float32.
            self.state_grid = np.zeros((t, e, N_SHARED_STATE_CHANNELS, h, w), dtype=np.float32)
            self.agent_state = np.zeros((t, e, n, N_AGENT_STATE_CHANNELS, h, w), dtype=np.uint8)
            self.state_vec = np.zeros((t, e, N_GLOBAL_VEC), dtype=np.float32)

        self._write_idx = 0

    @property
    def stores_global_state(self) -> bool:
        return self.state_hw is not None

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
        agent_state: np.ndarray | None = None,
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

        if self.state_grid is not None and self.agent_state is not None and self.state_vec is not None:
            if state_grid is None or state_vec is None or agent_state is None:
                raise ValueError(
                    "this buffer stores global state; state_grid, state_vec and agent_state are required"
                )
            self.state_grid[t] = state_grid
            self.state_vec[t] = state_vec
            self.agent_state[t] = agent_state

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

        def opt(x: np.ndarray | None, as_float: bool = False) -> torch.Tensor | None:
            if x is None:
                return None
            tensor = torch.as_tensor(x, device=dev)
            return tensor.float() if as_float else tensor

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
            state_grid=opt(self.state_grid),
            agent_state=opt(self.agent_state, as_float=True),
            state_vec=opt(self.state_vec),
        )


@dataclass
class RolloutBufferTensors:
    """Torch view of a `RolloutBuffer`, all still `[T, E, N, ...]` (shared
    state `[T, E, ...]`)."""

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
    agent_state: torch.Tensor | None = None
    state_vec: torch.Tensor | None = None

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated


@dataclass
class FlatBatch:
    """Flattened `[B, ...]` samples (B = T*E*N), post-GAE, for minibatching.
    `state_grid` here is the FULL 9-channel agent-specific critic grid."""

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


def build_agent_specific_state(
    shared_grid: torch.Tensor,  # [..., 7, H, W] — leading dims WITHOUT the agent axis
    agent_state: torch.Tensor,  # [..., N, 2, H, W]
    state_vec: torch.Tensor,  # [..., 4]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the agent-specific critic input (§9.3): the shared channels
    broadcast to every agent, concatenated with that agent's own 2 channels.
    Returns (grid[..., N, 9, H, W], vec[..., N, 4]). Used identically at
    rollout time, at the truncation bootstrap, and at update time, so the
    critic always sees the same channel layout."""
    n = agent_state.shape[-4]
    shared = shared_grid.unsqueeze(-4).expand(*shared_grid.shape[:-3], n, *shared_grid.shape[-3:])
    grid = torch.cat([shared, agent_state], dim=-3)
    vec = state_vec.unsqueeze(-2).expand(*state_vec.shape[:-1], n, state_vec.shape[-1])
    return grid, vec


def flatten_batch(
    buf: RolloutBufferTensors,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> FlatBatch:
    """Flatten [T,E,N,...] -> [T*E*N,...] for minibatch sampling (§16.5).
    Flattening happens only AFTER GAE (§16.8 item 2)."""

    def flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *x.shape[3:]) if x.dim() > 3 else x.reshape(-1)

    state_grid: torch.Tensor | None = None
    state_vec: torch.Tensor | None = None
    if buf.state_grid is not None and buf.agent_state is not None and buf.state_vec is not None:
        grid, vec = build_agent_specific_state(buf.state_grid, buf.agent_state, buf.state_vec)
        state_grid = flat(grid)  # [T,E,N,9,H,W] -> [B,9,H,W]
        state_vec = vec.reshape(-1, vec.shape[-1])  # [T,E,N,4] -> [B,4]

    return FlatBatch(
        obs_grid=flat(buf.obs_grid),
        obs_vec=flat(buf.obs_vec),
        mask=flat(buf.mask),
        action=flat(buf.action),
        logp_old=flat(buf.logp_old),
        advantages=flat(advantages),
        returns=flat(returns),
        active=flat(buf.active),
        state_grid=state_grid,
        state_vec=state_vec,
    )

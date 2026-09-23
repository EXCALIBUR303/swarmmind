"""SwarmParallelEnv: the PettingZoo Parallel API wrapper over SwarmWorld (§7.1-7.2).

Thin adapter only — all dynamics, observations, rewards and global state
come from `marl.envs.core.world.SwarmWorld`. This wrapper's only job is
dict-API bookkeeping (agent names, per-agent spaces, episode-end handling).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.observations import N_OBS_CHANNELS, N_VEC_FEATURES
from marl.envs.core.types import N_ACTIONS
from marl.envs.core.world import SwarmWorld

# state() returns only the team-SHARED portion of the global state (§9.1
# channels 0-6) — it has no per-agent context, so the two agent-specific
# channels (7-8) don't apply here. The full 9-channel agent-specific grid
# used by the MAPPO critic is built separately via
# `SwarmWorld.full_global_grid(i)`, not through this PettingZoo `state()`.
N_SHARED_STATE_CHANNELS = 7


def _agent_name(i: int) -> str:
    return f"robot_{i}"


class SwarmParallelEnv(ParallelEnv):
    """A PettingZoo `ParallelEnv` view of one `SwarmWorld` instance."""

    metadata = {"render_modes": ["ansi"], "name": "swarmmind_v0"}

    def __init__(self, config: EnvConfig, render_mode: str | None = None) -> None:
        self.config = config
        self.render_mode = render_mode
        self.world = SwarmWorld(config)

        self.possible_agents = [_agent_name(i) for i in range(config.n_robots)]
        self.agents: list[str] = []
        # Only used when reset(seed=None) — PettingZoo's own API tests call
        # reset() without a seed and expect the env to self-randomize.
        self._auto_seed_rng = np.random.default_rng()

        k = 2 * config.obs_radius + 1
        self._obs_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(low=0.0, high=1.0, shape=(N_OBS_CHANNELS, k, k), dtype=np.float32),
                "vec": gym.spaces.Box(low=0.0, high=1.0, shape=(N_VEC_FEATURES,), dtype=np.float32),
                "action_mask": gym.spaces.MultiBinary(N_ACTIONS),
            }
        )
        self._act_space = gym.spaces.Discrete(N_ACTIONS)
        self._state_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(N_SHARED_STATE_CHANNELS, config.height, config.width), dtype=np.float32
        )

    # -- Spaces -----------------------------------------------------------

    def observation_space(self, agent: str) -> gym.Space:
        return self._obs_space

    def action_space(self, agent: str) -> gym.Space:
        return self._act_space

    @property
    def state_space(self) -> gym.Space:
        return self._state_space

    # -- Lifecycle ----------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        options = options or {}
        map_spec: MapSpec | None = options.get("map")
        if seed is None and map_spec is None:
            seed = int(self._auto_seed_rng.integers(0, 2**31 - 1))
        self.world.reset(seed=seed, map_spec=map_spec)
        self.agents = list(self.possible_agents)

        grids, vecs, masks = self.world.all_actor_obs()
        obs = {
            name: {"grid": grids[i], "vec": vecs[i], "action_mask": masks[i]}
            for i, name in enumerate(self.agents)
        }
        infos: dict[str, dict[str, Any]] = {name: {} for name in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError("step() called with no active agents; call reset() first")

        action_arr = np.array([actions[name] for name in self.agents], dtype=np.int32)
        result = self.world.step(action_arr)

        grids, vecs, masks = self.world.all_actor_obs()
        obs = {
            name: {"grid": grids[i], "vec": vecs[i], "action_mask": masks[i]}
            for i, name in enumerate(self.agents)
        }
        rewards = {name: float(result.reward[i]) for i, name in enumerate(self.agents)}
        terminations = {name: result.terminated for name in self.agents}
        truncations = {name: result.truncated for name in self.agents}
        infos = result.info_dict(self.agents)

        if result.terminated or result.truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Full global state array (§7.2) — TRAINING/analysis use only, never fed to actors."""
        shared_grid, _ = self.world.global_state()
        return shared_grid

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return self.world.render_ansi()
        return None

    def close(self) -> None:
        pass

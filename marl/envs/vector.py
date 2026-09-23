"""SyncVectorSwarm: batched, auto-resetting vector wrapper over SwarmWorld (§7.3).

Used by training and evaluation for speed — steps `E` independent
`SwarmWorld` copies per call instead of converting to/from PettingZoo dicts
every step. Thin adapter only; all dynamics come from `SwarmWorld`.

Auto-reset semantics: when env slot `e` finishes (terminated or
truncated), its final observation/state are captured into `Batch.final`
BEFORE the slot is reset (needed to bootstrap truncated episodes, §15.6).
The `obs`/`state` fields of the returned `Batch` for slot `e` are the
NEW episode's first observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.envs.core.config import EnvConfig
from marl.envs.core.world import SwarmWorld
from marl.utils.seeding import SeedStreams

MAX_SEED = 2**31 - 1


@dataclass
class FinalInfo:
    """Captured just before an auto-reset, one entry per env slot that ended
    this call (§15.6 truncation bootstrap needs the TRUE final observation,
    not the post-reset one)."""

    env_indices: list[int]
    obs_grid: np.ndarray  # [K_ended, N, C, K, K]
    obs_vec: np.ndarray  # [K_ended, N, D]
    state_grid: np.ndarray  # [K_ended, Cg, H, W]  (shared channels only)
    state_vec: np.ndarray  # [K_ended, 4]
    episode_metrics: list[dict]  # one dict per ended env: t, coverage, n_rescued, n_discovered, success


@dataclass
class Batch:
    obs_grid: np.ndarray  # float32[E, N, C, K, K]
    obs_vec: np.ndarray  # float32[E, N, D]
    mask: np.ndarray  # int8[E, N, 5]
    state_grid: np.ndarray  # float32[E, 7, H, W] — shared channels only (agent-specific built by the trainer)
    state_vec: np.ndarray  # float32[E, 4]
    reward: np.ndarray  # float32[E, N]
    terminated: np.ndarray  # bool[E]
    truncated: np.ndarray  # bool[E]
    final: FinalInfo | None


class SyncVectorSwarm:
    """`n_envs` independent `SwarmWorld` copies, stepped synchronously."""

    def __init__(
        self,
        config: EnvConfig,
        n_envs: int,
        root_seed: int | None = None,
        seed_generators: list[np.random.Generator] | None = None,
    ) -> None:
        if seed_generators is None:
            if root_seed is None:
                raise ValueError("SyncVectorSwarm requires either root_seed or seed_generators")
            seed_generators = SeedStreams(root_seed).spawn_generators(n_envs)
        if len(seed_generators) != n_envs:
            raise ValueError(f"expected {n_envs} seed generators, got {len(seed_generators)}")

        self.config = config
        self.n_envs = n_envs
        self.n_robots = config.n_robots
        self._seed_gens = seed_generators
        self.worlds = [SwarmWorld(config) for _ in range(n_envs)]
        self._reset_done = False

    def _next_seed(self, e: int) -> int:
        return int(self._seed_gens[e].integers(0, MAX_SEED))

    def reset(self) -> Batch:
        for e, w in enumerate(self.worlds):
            w.reset(seed=self._next_seed(e))
        self._reset_done = True
        obs_grid, obs_vec, mask = self._collect_obs()
        state_grid, state_vec = self._collect_state()
        n = self.n_robots
        return Batch(
            obs_grid=obs_grid,
            obs_vec=obs_vec,
            mask=mask,
            state_grid=state_grid,
            state_vec=state_vec,
            reward=np.zeros((self.n_envs, n), dtype=np.float32),
            terminated=np.zeros(self.n_envs, dtype=bool),
            truncated=np.zeros(self.n_envs, dtype=bool),
            final=None,
        )

    def step(self, actions: np.ndarray) -> Batch:
        if not self._reset_done:
            raise RuntimeError("step() called before reset()")
        actions = np.asarray(actions)
        if actions.shape != (self.n_envs, self.n_robots):
            raise ValueError(
                f"actions must have shape ({self.n_envs}, {self.n_robots}), got {actions.shape}"
            )

        reward = np.zeros((self.n_envs, self.n_robots), dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)

        ended_indices: list[int] = []
        ended_obs_grid: list[np.ndarray] = []
        ended_obs_vec: list[np.ndarray] = []
        ended_state_grid: list[np.ndarray] = []
        ended_state_vec: list[np.ndarray] = []
        ended_metrics: list[dict] = []

        for e, w in enumerate(self.worlds):
            result = w.step(actions[e])
            reward[e] = result.reward
            terminated[e] = result.terminated
            truncated[e] = result.truncated

            if result.terminated or result.truncated:
                og, ov, _ = w.all_actor_obs()
                sg, sv = w.global_state()
                ended_indices.append(e)
                ended_obs_grid.append(og)
                ended_obs_vec.append(ov)
                ended_state_grid.append(sg)
                ended_state_vec.append(sv)
                ended_metrics.append(
                    {
                        "t": result.t,
                        "coverage": result.coverage,
                        "n_rescued": result.n_rescued,
                        "n_discovered": result.n_discovered,
                        "success": result.success,
                    }
                )
                w.reset(seed=self._next_seed(e))

        obs_grid, obs_vec, mask = self._collect_obs()
        state_grid, state_vec = self._collect_state()

        final = None
        if ended_indices:
            final = FinalInfo(
                env_indices=ended_indices,
                obs_grid=np.stack(ended_obs_grid),
                obs_vec=np.stack(ended_obs_vec),
                state_grid=np.stack(ended_state_grid),
                state_vec=np.stack(ended_state_vec),
                episode_metrics=ended_metrics,
            )

        return Batch(
            obs_grid=obs_grid,
            obs_vec=obs_vec,
            mask=mask,
            state_grid=state_grid,
            state_vec=state_vec,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            final=final,
        )

    def _collect_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grids, vecs, masks = [], [], []
        for w in self.worlds:
            g, v, m = w.all_actor_obs()
            grids.append(g)
            vecs.append(v)
            masks.append(m)
        return np.stack(grids), np.stack(vecs), np.stack(masks)

    def _collect_state(self) -> tuple[np.ndarray, np.ndarray]:
        grids, vecs = [], []
        for w in self.worlds:
            g, v = w.global_state()
            grids.append(g)
            vecs.append(v)
        return np.stack(grids), np.stack(vecs)

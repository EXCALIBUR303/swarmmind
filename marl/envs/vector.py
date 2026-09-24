"""SyncVectorSwarm: batched, auto-resetting vector wrapper over SwarmWorld (§7.3).

Used by training and evaluation for speed — steps `E` independent
`SwarmWorld` copies per call instead of converting to/from PettingZoo dicts
every step. Thin adapter only; all dynamics come from `SwarmWorld`.

Auto-reset semantics: when env slot `e` finishes (terminated or
truncated), its final observation/state are captured into `Batch.final`
BEFORE the slot is reset (needed to bootstrap truncated episodes, §15.6).
The `obs`/`state` fields of the returned `Batch` for slot `e` are the
NEW episode's first observation.

`exclude_hashes` is the training-isolation guard (§20.3): any generated
map whose content hash is in the set (the frozen validation + test maps)
is rejected and a fresh seed drawn, so no evaluation map can ever be
trained on. Rejections are counted in `rejected_maps` for auditing.

`include_agent_state=True` additionally returns the two agent-specific
global-state channels per robot (§9.1 channels 7-8: own position, own
belief-known mask), which only the MAPPO centralized critic needs. It is
opt-in so IPPO doesn't pay for them; computing them consumes no RNG, so
turning it on never changes a training trajectory (checked by the
IPPO/MAPPO parity test, §16.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.envs.core.config import EnvConfig
from marl.envs.core.mapgen import generate_map
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
    state_grid: np.ndarray  # [K_ended, 7, H, W]  (shared channels only)
    state_vec: np.ndarray  # [K_ended, 4]
    episode_metrics: list[dict]  # one dict per ended env: t, coverage, n_rescued, n_discovered, success
    agent_state: np.ndarray | None = None  # [K_ended, N, 2, H, W] when include_agent_state


@dataclass
class Batch:
    obs_grid: np.ndarray  # float32[E, N, C, K, K]
    obs_vec: np.ndarray  # float32[E, N, D]
    mask: np.ndarray  # int8[E, N, 5]
    state_grid: np.ndarray  # float32[E, 7, H, W] — shared channels only
    state_vec: np.ndarray  # float32[E, 4]
    reward: np.ndarray  # float32[E, N]
    terminated: np.ndarray  # bool[E]
    truncated: np.ndarray  # bool[E]
    final: FinalInfo | None
    agent_state: np.ndarray | None = None  # float32[E, N, 2, H, W] when include_agent_state


def _agent_state(world: SwarmWorld) -> np.ndarray:
    """[N, 2, H, W] agent-specific channels for every robot in `world`."""
    assert world.state is not None
    return np.stack([world.agent_state_channels(i) for i in range(world.state.n_robots)])


class SyncVectorSwarm:
    """`n_envs` independent `SwarmWorld` copies, stepped synchronously."""

    def __init__(
        self,
        config: EnvConfig,
        n_envs: int,
        root_seed: int | None = None,
        seed_generators: list[np.random.Generator] | None = None,
        include_agent_state: bool = False,
        exclude_hashes: frozenset[str] | None = None,
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
        self.include_agent_state = include_agent_state
        self.exclude_hashes = exclude_hashes or frozenset()
        self.rejected_maps = 0
        self._seed_gens = seed_generators
        self.worlds = [SwarmWorld(config) for _ in range(n_envs)]
        self._reset_done = False

    def _next_seed(self, e: int) -> int:
        return int(self._seed_gens[e].integers(0, MAX_SEED))

    def _reset_world(self, e: int) -> None:
        """Reset env slot `e` onto a fresh training map from its seed
        stream, skipping any map in `exclude_hashes`. With no collision this
        is exactly `world.reset(seed=...)` (same generator, same map id)."""
        while True:
            spec = generate_map(self.config, self._next_seed(e))
            if self.exclude_hashes and spec.content_hash in self.exclude_hashes:
                self.rejected_maps += 1
                continue
            self.worlds[e].reset(map_spec=spec)
            return

    def _make_batch(
        self,
        reward: np.ndarray | None = None,
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
        final: FinalInfo | None = None,
    ) -> Batch:
        obs_grid, obs_vec, mask = self._collect_obs()
        state_grid, state_vec = self._collect_state()
        agent_state = np.stack([_agent_state(w) for w in self.worlds]) if self.include_agent_state else None
        e, n = self.n_envs, self.n_robots
        return Batch(
            obs_grid=obs_grid,
            obs_vec=obs_vec,
            mask=mask,
            state_grid=state_grid,
            state_vec=state_vec,
            reward=reward if reward is not None else np.zeros((e, n), dtype=np.float32),
            terminated=terminated if terminated is not None else np.zeros(e, dtype=bool),
            truncated=truncated if truncated is not None else np.zeros(e, dtype=bool),
            final=final,
            agent_state=agent_state,
        )

    def current_batch(self) -> Batch:
        """Rebuild a `Batch` from each world's CURRENT state, without
        stepping or resetting anything. Used after directly restoring
        world states (e.g. training resume, §33.3) so the trainer's next
        rollout observation reflects the restored states rather than
        whatever `reset()` last produced."""
        if not self._reset_done:
            raise RuntimeError("current_batch() called before reset()")
        return self._make_batch()

    def reset(self) -> Batch:
        for e in range(self.n_envs):
            self._reset_world(e)
        self._reset_done = True
        return self._make_batch()

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
        ended_agent_state: list[np.ndarray] = []
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
                if self.include_agent_state:
                    ended_agent_state.append(_agent_state(w))
                ended_metrics.append(
                    {
                        "t": result.t,
                        "coverage": result.coverage,
                        "n_rescued": result.n_rescued,
                        "n_discovered": result.n_discovered,
                        "success": result.success,
                    }
                )
                self._reset_world(e)

        final = None
        if ended_indices:
            final = FinalInfo(
                env_indices=ended_indices,
                obs_grid=np.stack(ended_obs_grid),
                obs_vec=np.stack(ended_obs_vec),
                state_grid=np.stack(ended_state_grid),
                state_vec=np.stack(ended_state_vec),
                episode_metrics=ended_metrics,
                agent_state=np.stack(ended_agent_state) if self.include_agent_state else None,
            )

        return self._make_batch(reward, terminated, truncated, final)

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

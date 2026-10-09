"""Episode runner: drives a `Policy` through a `SwarmWorld` episode and
produces an `EpisodeTrace` (§27.1) plus, optionally, aggregated metrics.

This is the SINGLE place an episode is "run" for evaluation purposes —
training uses the vectorized wrapper directly (§7.3) for speed, but
evaluation, replay recording, and the backend's live sessions should all
go through (or match) this function, per the single-code-path principle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.algorithms.common.policy import Policy
from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.types import OBSTACLE, EpisodeTrace, StepResult
from marl.envs.core.world import SwarmWorld


@dataclass
class EpisodeResult:
    trace: EpisodeTrace
    n_reachable_free: int


class EpisodeRunner:
    """Steps ONE episode and keeps its `EpisodeTrace`.

    The single implementation of "advance an episode and record it":
    `run_episode` (evaluation) drives it to completion, and the backend's
    live sessions (§28.4) drive it one step at a time — so a live session
    cannot drift from the evaluator, by construction.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        seed: int | None = None,
        map_spec: MapSpec | None = None,
    ) -> None:
        self.cfg = cfg
        self.world = SwarmWorld(cfg)
        self.state = self.world.reset(seed=seed, map_spec=map_spec)
        self.n_reachable_free = int(np.sum(self.state.terrain != OBSTACLE))
        self.trace = EpisodeTrace(
            map_id=self.state.map_id,
            width=cfg.width,
            height=cfg.height,
            n_robots=cfg.n_robots,
            n_targets=cfg.n_targets,
        )
        self.trace.positions.append(self.state.robot_pos.copy())
        self.trace.coverage_over_time.append(
            float(np.sum(self.state.known_team & (self.state.terrain != OBSTACLE)))
            / max(self.n_reachable_free, 1)
        )
        self.done = False

    def step(self, policy: Policy, rng: np.random.Generator) -> tuple[np.ndarray, StepResult]:
        """Advance one joint step; returns `(actions, StepResult)`."""
        if self.done:
            raise RuntimeError("episode already finished")
        actions = policy.act(self.world, rng)
        result = self.world.step(actions)
        trace = self.trace
        trace.positions.append(self.state.robot_pos.copy())
        trace.actions.append(actions.copy())
        trace.rewards.append(result.reward.copy())
        trace.reward_components.append({k: v.copy() for k, v in result.reward_components.items()})
        trace.moved.append(result.moved.copy())
        trace.blocked_by_robot.append(result.blocked_by_robot.copy())
        trace.bumped_wall.append(result.bumped_wall.copy())
        trace.discovered_events.append(list(result.discovered))
        trace.rescued_events.append(list(result.rescued))
        trace.coverage_over_time.append(result.coverage)

        if result.terminated or result.truncated:
            trace.success = result.success
            trace.final_t = result.t
            self.done = True
        elif len(trace.actions) >= self.cfg.max_steps:
            trace.success = False
            trace.final_t = self.cfg.max_steps
            self.done = True
        return actions, result


def run_episode(
    cfg: EnvConfig,
    policy: Policy,
    rng: np.random.Generator,
    seed: int | None = None,
    map_spec: MapSpec | None = None,
) -> EpisodeResult:
    """Run one full episode of `policy` on `cfg`, returning its trace.

    Exactly one of `seed`/`map_spec` should be meaningful (matching
    `SwarmWorld.reset`'s contract); `rng` drives the POLICY's own
    stochasticity (e.g. Random's action sampling), not map generation.
    """
    runner = EpisodeRunner(cfg, seed=seed, map_spec=map_spec)
    while not runner.done:
        runner.step(policy, rng)
    return EpisodeResult(trace=runner.trace, n_reachable_free=runner.n_reachable_free)

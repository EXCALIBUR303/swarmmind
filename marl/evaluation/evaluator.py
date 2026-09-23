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
from marl.envs.core.types import OBSTACLE, EpisodeTrace
from marl.envs.core.world import SwarmWorld


@dataclass
class EpisodeResult:
    trace: EpisodeTrace
    n_reachable_free: int


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
    world = SwarmWorld(cfg)
    state = world.reset(seed=seed, map_spec=map_spec)

    n_reachable_free = int(np.sum(state.terrain != OBSTACLE))
    n_free = max(n_reachable_free, 1)

    trace = EpisodeTrace(
        map_id=state.map_id,
        width=cfg.width,
        height=cfg.height,
        n_robots=cfg.n_robots,
        n_targets=cfg.n_targets,
    )
    trace.positions.append(state.robot_pos.copy())
    trace.coverage_over_time.append(
        float(np.sum(state.known_team & (state.terrain != OBSTACLE))) / n_free
    )

    for _ in range(cfg.max_steps):
        actions = policy.act(world, rng)
        result = world.step(actions)

        trace.positions.append(state.robot_pos.copy())
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
            break
    else:
        trace.success = False
        trace.final_t = cfg.max_steps

    return EpisodeResult(trace=trace, n_reachable_free=n_reachable_free)

"""Replay recorder (§27.1): wraps an episode run and emits a `Replay`.

Frames come directly from `SwarmWorld`'s own `StepResult` — never
re-simulated or reconstructed after the fact — so a replay always matches
what actually happened during evaluation (§5.3 single code path).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from marl.algorithms.common.policy import Policy
from marl.analysis.metrics import compute_episode_metrics
from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.types import OBSTACLE, EpisodeTrace
from marl.envs.core.world import SwarmWorld
from marl.replay.schema import (
    Replay,
    ReplayEnvInfo,
    ReplayFrame,
    ReplayMapInfo,
    ReplayOutcome,
    ReplayPolicyInfo,
)


def record_episode(
    cfg: EnvConfig,
    policy: Policy,
    rng: np.random.Generator,
    policy_info: ReplayPolicyInfo,
    replay_id: str,
    seed: int | None = None,
    map_spec: MapSpec | None = None,
) -> Replay:
    """Run one episode of `policy` and record it as a `Replay`."""
    world = SwarmWorld(cfg)
    state = world.reset(seed=seed, map_spec=map_spec)

    # Rebuild the exact MapSpec (terrain + starts + targets) from world state
    # so the replay's map info and content hash always match what was
    # actually played, regardless of whether it came from a seed or an
    # explicit map_spec.
    played_map_spec = MapSpec.from_terrain_array(
        id=state.map_id,
        terrain=state.terrain,
        robot_starts=[(int(p[0]), int(p[1])) for p in state.robot_pos],
        targets=[(int(p[0]), int(p[1])) for p in state.target_pos],
    )
    map_info = ReplayMapInfo(
        id=state.map_id,
        hash=played_map_spec.content_hash,
        width=state.width,
        height=state.height,
        terrain=played_map_spec.terrain_rows,
        robot_starts=[[int(p[0]), int(p[1])] for p in state.robot_pos],
        targets=[[int(p[0]), int(p[1])] for p in state.target_pos],
    )

    trace = EpisodeTrace(
        map_id=state.map_id, width=cfg.width, height=cfg.height, n_robots=cfg.n_robots, n_targets=cfg.n_targets
    )
    trace.positions.append(state.robot_pos.copy())
    n_reachable_free = int(np.sum(state.terrain != OBSTACLE))
    trace.coverage_over_time.append(
        float(np.sum(state.known_team & (state.terrain != OBSTACLE))) / max(n_reachable_free, 1)
    )

    frames: list[ReplayFrame] = []
    known_before = state.known_team.copy()

    for _ in range(cfg.max_steps):
        actions = policy.act(world, rng)
        result = world.step(actions)

        known_after = state.known_team
        newly_known_cells = np.argwhere(known_after & ~known_before).tolist()
        known_before = known_after.copy()

        frames.append(
            ReplayFrame(
                t=result.t,
                pos=state.robot_pos.tolist(),
                action=actions.tolist(),
                moved=result.moved.tolist(),
                blocked_by_robot=result.blocked_by_robot.tolist(),
                bumped_wall=result.bumped_wall.tolist(),
                newly_known=newly_known_cells,
                discovered=list(result.discovered),
                rescued=[[k, r] for k, r in result.rescued],
                reward=result.reward.tolist(),
                reward_components={k: v.tolist() for k, v in result.reward_components.items()},
            )
        )

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

    metrics = compute_episode_metrics(trace, n_reachable_free)

    return Replay(
        replay_id=replay_id,
        created_at=datetime.now(UTC).isoformat(),
        env=ReplayEnvInfo(level=cfg.name, config=cfg.model_dump()),
        map=map_info,
        policy=policy_info,
        frames=frames,
        outcome=ReplayOutcome(
            success=trace.success,
            steps=trace.final_t,
            metrics={
                "episode_return_mean": metrics.episode_return_mean,
                "coverage_final": metrics.coverage_final,
                "n_discovered": metrics.n_discovered,
                "n_rescued": metrics.n_rescued,
                "n_collisions": metrics.n_collisions,
            },
        ),
    )

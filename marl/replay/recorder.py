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
from marl.envs.core.types import StepResult, WorldState
from marl.replay.schema import (
    Replay,
    ReplayEnvInfo,
    ReplayFrame,
    ReplayInitial,
    ReplayMapInfo,
    ReplayOutcome,
    ReplayPolicyInfo,
)


def build_frame(
    state: WorldState,
    actions: np.ndarray,
    result: StepResult,
    known_before: np.ndarray,
) -> tuple[ReplayFrame, np.ndarray]:
    """One step's `ReplayFrame` (§27.2), taken straight from the world's own
    `StepResult`. Returns `(frame, known_after_copy)`; pass the latter as
    `known_before` for the next step. Shared by the recorder and the
    backend's live sessions so both emit byte-identical frames."""
    known_after = state.known_team
    newly_known_cells = np.argwhere(known_after & ~known_before).tolist()
    frame = ReplayFrame(
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
    return frame, known_after.copy()


def record_episode(
    cfg: EnvConfig,
    policy: Policy,
    rng: np.random.Generator,
    policy_info: ReplayPolicyInfo,
    replay_id: str,
    seed: int | None = None,
    map_spec: MapSpec | None = None,
) -> Replay:
    """Run one episode of `policy` and record it as a `Replay`. Steps through `EpisodeRunner`, the
    same stepper the evaluator and the backend's live sessions use, so the trace (and therefore the
    metrics) cannot differ between them."""
    from marl.evaluation.evaluator import EpisodeRunner

    runner = EpisodeRunner(cfg, seed=seed, map_spec=map_spec)
    state = runner.state

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

    frames: list[ReplayFrame] = []
    known_before = state.known_team.copy()
    initial_known = np.argwhere(known_before).tolist()
    while not runner.done:
        actions, result = runner.step(policy, rng)
        frame, known_before = build_frame(state, actions, result, known_before)
        frames.append(frame)
    trace, n_reachable_free = runner.trace, runner.n_reachable_free

    metrics = compute_episode_metrics(trace, n_reachable_free)

    return Replay(
        replay_id=replay_id,
        created_at=datetime.now(UTC).isoformat(),
        env=ReplayEnvInfo(level=cfg.name, config=cfg.model_dump()),
        map=map_info,
        policy=policy_info,
        initial=ReplayInitial(known_team=initial_known, discovered=list(trace.initial_discovered)),
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

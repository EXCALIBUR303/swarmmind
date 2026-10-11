"""Replay file schema (§27.2), pydantic models exported as JSON.

Phase 3 scope: the recorder writes replays for baseline (Random/heuristic)
episodes. Learned-policy metadata (`checkpoint_step`, `run_id`) is wired
up starting Phase 4/5 when those artifacts exist.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class ReplayEnvInfo(BaseModel):
    level: str
    config: dict[str, Any]


class ReplayMapInfo(BaseModel):
    id: str
    hash: str
    width: int
    height: int
    terrain: list[str]
    robot_starts: list[list[int]]
    targets: list[list[int]]


class ReplayPolicyInfo(BaseModel):
    kind: Literal["random", "heuristic", "ippo", "mappo"]
    run_id: str | None = None
    checkpoint_step: int | None = None
    stochastic: bool = True
    eval_seed: int | None = None


class ReplayInitial(BaseModel):
    """State right after reset (§27.2 `initial`). The reset-time sensing pass
    reveals cells that no frame's `newly_known` delta ever contains, so a
    viewer rebuilding the known map needs them: known(t) = initial.known_team
    ∪ ⋃_{i<=t} frames[i].newly_known."""

    known_team: list[list[int]]
    # Targets already sensed at reset (Phase 15 fix): they appear in no frame's `discovered`.
    # Optional so replays recorded before the field existed still validate (read as "none").
    discovered: list[int] = Field(default_factory=list)


class ReplayFrame(BaseModel):
    """One step's delta (§27.2). `t` is the POST-step index (1-based)."""

    t: int
    pos: list[list[int]]
    action: list[int]
    moved: list[bool]
    blocked_by_robot: list[bool]
    bumped_wall: list[bool]
    newly_known: list[list[int]] = Field(default_factory=list)
    discovered: list[int] = Field(default_factory=list)
    rescued: list[list[int]] = Field(default_factory=list)  # [[target_idx, robot_idx], ...]
    reward: list[float]
    reward_components: dict[str, list[float]]
    comm_edges: list[list[int]] = Field(default_factory=list)


class ReplayOutcome(BaseModel):
    success: bool
    steps: int
    metrics: dict[str, Any]


class Replay(BaseModel):
    schema_version: int = SCHEMA_VERSION
    replay_id: str
    created_at: str
    env: ReplayEnvInfo
    map: ReplayMapInfo
    policy: ReplayPolicyInfo
    # Absent in replays recorded before this field existed (Phases 3-8); the
    # backend derives it for those by re-running reset (GET /replays/{id}/initial).
    initial: ReplayInitial | None = None
    frames: list[ReplayFrame]
    outcome: ReplayOutcome

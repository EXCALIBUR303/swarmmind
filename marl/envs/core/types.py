"""Core dataclasses and constants for the SwarmWorld environment (§6, §7.1).

These types are the shared vocabulary between `world.py`, the PettingZoo
wrapper, the vector wrapper, the replay recorder and the analysis code. No
torch or algorithm imports belong here (see the import-boundary rules in
`tests/test_import_boundaries.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --- Terrain -----------------------------------------------------------
FREE = 0
OBSTACLE = 1

# --- Target status (§6.2) -----------------------------------------------
HIDDEN = 0
DISCOVERED = 1
RESCUED = 2

# --- Actions (§10.1) -----------------------------------------------------
WAIT = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
N_ACTIONS = 5

# (dy, dx) for each action index.
ACTION_DELTAS: np.ndarray = np.array(
    [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int32
)

# --- Communication modes (§19) -------------------------------------------
COMM_NONE = "none"
COMM_SHARED_MAP = "shared_map"
COMM_LIMITED_RANGE = "limited_range"


@dataclass
class RobotBelief:
    """What one robot is entitled to know (§11.4, §12.2).

    `terrain_belief` uses -1=unknown, FREE=0, OBSTACLE=1. `known` is
    `terrain_belief >= 0` and is kept as a derived property rather than a
    separate array to avoid the two ever going out of sync.

    `target_known[k]` / `target_rescued_known[k]` track what this robot
    knows about target k. Since targets are static, a remembered position
    stays valid forever once known — but rescue status must still be
    tracked per belief so that mode A (no communication) can lag behind
    the true global state.
    """

    terrain_belief: np.ndarray  # int8[H, W], -1 unknown / 0 free / 1 obstacle
    target_known: np.ndarray  # bool[M]
    target_rescued_known: np.ndarray  # bool[M]
    visited_self: np.ndarray  # bool[H, W]
    visited_team_known: np.ndarray  # bool[H, W] — cells known to have been visited by ANY robot

    @property
    def known(self) -> np.ndarray:
        return self.terrain_belief >= 0

    def copy(self) -> RobotBelief:
        return RobotBelief(
            terrain_belief=self.terrain_belief.copy(),
            target_known=self.target_known.copy(),
            target_rescued_known=self.target_rescued_known.copy(),
            visited_self=self.visited_self.copy(),
            visited_team_known=self.visited_team_known.copy(),
        )


@dataclass
class WorldState:
    """A snapshot of everything SwarmWorld tracks (§6.2).

    Returned by `SwarmWorld.reset()`. Arrays are the SAME objects the
    world mutates internally (not copies) for efficiency — callers that
    need a stable snapshot (e.g. replay recording) must copy explicitly.
    """

    width: int
    height: int
    n_robots: int
    n_targets: int
    terrain: np.ndarray  # int8[H, W]
    robot_pos: np.ndarray  # int32[N, 2] (y, x)
    target_pos: np.ndarray  # int32[M, 2] (y, x)
    target_status: np.ndarray  # int8[M]
    target_rescued_by: np.ndarray  # int32[M], -1 if not rescued
    target_t_discovered: np.ndarray  # int32[M], -1 if not yet discovered
    target_t_rescued: np.ndarray  # int32[M], -1 if not yet rescued
    known_team: np.ndarray  # bool[H, W] — cells known to the team (union of all sensing)
    visited_count: np.ndarray  # int32[H, W] — cumulative robot-visits per cell (global state ch.6)
    beliefs: list[RobotBelief]
    t: int
    map_id: str
    prev_action: np.ndarray  # int32[N], -1 before the first step
    last_blocked: np.ndarray  # bool[N] — blocked_by_robot | bumped_wall on the most recent step


@dataclass
class StepResult:
    """Everything `SwarmWorld.step()` returns for one joint step (§6.4).

    Observations are NOT included here — callers fetch them separately via
    `actor_obs`/`all_actor_obs`/`global_state`, per the single-code-path
    design (spec §7.1, §5.3).
    """

    reward: np.ndarray  # float32[N]
    reward_components: dict[str, np.ndarray]  # each float32[N] (or a scalar-broadcast constant)
    terminated: bool
    truncated: bool
    moved: np.ndarray  # bool[N]
    blocked_by_robot: np.ndarray  # bool[N]
    bumped_wall: np.ndarray  # bool[N]
    rescued: list[tuple[int, int]]  # (target_idx, robot_idx) rescued this step
    discovered: list[int]  # target indices newly discovered this step
    newly_known_count: int  # |K_{t+1}| - |K_t|, team cells newly known this step
    t: int  # step index AFTER this step (i.e. world.t post-increment)
    coverage: float  # |known_team ∩ free| / |free|
    n_discovered: int
    n_rescued: int
    success: bool

    def info_dict(self, agent_names: list[str]) -> dict[str, dict[str, Any]]:
        """Build the per-agent `infos` dict for the PettingZoo wrapper (§7.2)."""
        team_info = {
            "t": self.t,
            "coverage": self.coverage,
            "n_discovered": self.n_discovered,
            "n_rescued": self.n_rescued,
            "success": self.success,
        }
        out: dict[str, dict[str, Any]] = {}
        for i, name in enumerate(agent_names):
            out[name] = {
                "moved": bool(self.moved[i]),
                "blocked_by_robot": bool(self.blocked_by_robot[i]),
                "bumped_wall": bool(self.bumped_wall[i]),
                "rescued": [k for k, r in self.rescued if r == i],
                "discovered": [k for k in self.discovered],  # team-level; same for all agents
                "explore_credit": float(self.reward_components.get("explore", np.zeros(len(agent_names)))[i]),
                "team": team_info,
            }
        return out


@dataclass
class EpisodeTrace:
    """Standard trace format consumed by `marl.analysis.metrics` and the
    replay recorder (§27.1), so both always compute from identical data.

    `positions[t]` is the joint position array (`int32[N, 2]`) BEFORE step
    t is applied; there are `T_ep + 1` position entries (including the
    initial reset state) and `T_ep` entries in every per-step list.
    """

    map_id: str
    width: int
    height: int
    n_robots: int
    n_targets: int
    positions: list[np.ndarray] = field(default_factory=list)  # len T_ep+1, each int32[N,2]
    actions: list[np.ndarray] = field(default_factory=list)  # len T_ep, each int32[N]
    rewards: list[np.ndarray] = field(default_factory=list)  # len T_ep, each float32[N]
    reward_components: list[dict[str, np.ndarray]] = field(default_factory=list)
    moved: list[np.ndarray] = field(default_factory=list)
    blocked_by_robot: list[np.ndarray] = field(default_factory=list)
    bumped_wall: list[np.ndarray] = field(default_factory=list)
    discovered_events: list[list[int]] = field(default_factory=list)  # per step
    rescued_events: list[list[tuple[int, int]]] = field(default_factory=list)  # per step
    coverage_over_time: list[float] = field(default_factory=list)
    success: bool = False
    final_t: int = 0

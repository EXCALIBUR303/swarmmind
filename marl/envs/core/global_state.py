"""Centralized critic global state (§9) — TRAINING ONLY.

This module deliberately exposes privileged information (ground-truth
terrain, hidden target positions, all robot positions) that the actor must
NEVER see. It exists purely for the MAPPO centralized critic (§16) and for
analysis/metrics that need ground truth. Nothing in `marl.networks.actor`
or `marl.algorithms.common.policy` may import this module — enforced by
`tests/test_import_boundaries.py` (§12.4 test_actor_import_boundary).
"""

from __future__ import annotations

import numpy as np

from marl.envs.core.types import OBSTACLE, RESCUED, WorldState

N_GLOBAL_CHANNELS = 9
N_GLOBAL_VEC_FEATURES = 4

# Shared channels (§9.1)
CH_OBSTACLE_TRUE = 0
CH_REACHABLE_FREE = 1
CH_TEAM_KNOWN = 2
CH_TARGETS_ALL = 3  # includes HIDDEN targets — privileged
CH_TARGETS_DISCOVERED = 4
CH_ROBOT_OCCUPANCY = 5
CH_VISIT_COUNT = 6
# Agent-specific channels (§9.1)
CH_AGENT_POS = 7
CH_AGENT_BELIEF_KNOWN = 8


def shared_global_grid(state: WorldState, visit_count_cap: int = 10) -> np.ndarray:
    """Channels 0-6, identical for every agent (§9.1)."""
    height, width = state.height, state.width
    grid = np.zeros((7, height, width), dtype=np.float32)

    grid[CH_OBSTACLE_TRUE] = (state.terrain == OBSTACLE).astype(np.float32)
    grid[CH_REACHABLE_FREE] = (state.terrain != OBSTACLE).astype(np.float32)
    grid[CH_TEAM_KNOWN] = state.known_team.astype(np.float32)

    for k in range(state.n_targets):
        if state.target_status[k] == RESCUED:
            continue
        y, x = state.target_pos[k]
        grid[CH_TARGETS_ALL, y, x] = 1.0
        if state.target_t_discovered[k] >= 0:
            grid[CH_TARGETS_DISCOVERED, y, x] = 1.0

    for i in range(state.n_robots):
        y, x = state.robot_pos[i]
        grid[CH_ROBOT_OCCUPANCY, y, x] = 1.0

    grid[CH_VISIT_COUNT] = np.minimum(state.visited_count.astype(np.float32) / visit_count_cap, 1.0)

    return grid


def shared_global_vec(state: WorldState, max_steps: int) -> np.ndarray:
    """4-dim global vector (§9.2)."""
    n_rescued = int(np.sum(state.target_status == RESCUED))
    n_discovered = int(np.sum(state.target_t_discovered >= 0))
    n_free = int(np.sum(state.terrain != OBSTACLE))
    coverage = float(np.sum(state.known_team & (state.terrain != OBSTACLE))) / max(n_free, 1)
    return np.array(
        [
            1.0 - (state.t / max_steps),
            n_rescued / max(state.n_targets, 1),
            coverage,
            n_discovered / max(state.n_targets, 1),
        ],
        dtype=np.float32,
    )


def agent_state_channels(state: WorldState, robot_idx: int) -> np.ndarray:
    """Channels 7-8, specific to `robot_idx` (§9.1)."""
    height, width = state.height, state.width
    grid = np.zeros((2, height, width), dtype=np.float32)
    y, x = state.robot_pos[robot_idx]
    grid[0, y, x] = 1.0  # CH_AGENT_POS (relative index within this 2-channel block)
    grid[1] = state.beliefs[robot_idx].known.astype(np.float32)  # CH_AGENT_BELIEF_KNOWN
    return grid


def full_global_grid(state: WorldState, robot_idx: int, visit_count_cap: int = 10) -> np.ndarray:
    """Convenience: the full 9-channel agent-specific global grid for one robot."""
    shared = shared_global_grid(state, visit_count_cap=visit_count_cap)
    agent = agent_state_channels(state, robot_idx)
    return np.concatenate([shared, agent], axis=0)

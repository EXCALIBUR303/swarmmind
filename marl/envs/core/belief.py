"""Per-robot belief construction and communication modes A/B/C (§11, §19).

`sense_and_update` performs step 7 ("Sense") of the step order in §6.4: for
every robot, compute what it senses this step, update its own belief, then
apply the communication mode's sharing rule.

Mode semantics (§19):
  - "none": beliefs never merge; each robot only ever learns from its own
    sensing.
  - "shared_map": after every robot senses, ALL map knowledge (terrain,
    target existence/rescued status, team-visited cells) is broadcast to
    every robot. Robot POSITIONS are never shared this way — teammates are
    only ever visible via current sensing (obs channel 6).
  - "limited_range": robots within `comm_radius` of each other (transitive
    over connected components of the comm graph) merge beliefs this step.
"""

from __future__ import annotations

import numpy as np

from marl.envs.core.sensing import sensed_cells
from marl.envs.core.types import DISCOVERED, HIDDEN, RESCUED, WorldState


def _connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = [False] * n
    components = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        components.append(comp)
    return components


def sense_and_update(
    state: WorldState,
    sensor_radius: float,
    comm_mode: str,
    comm_radius: float,
) -> tuple[list[int], np.ndarray]:
    """Perform sensing + belief update + communication sharing for one step.

    Mutates `state.beliefs` and `state.known_team` in place. Returns
    `(newly_discovered_target_indices, per_robot_sensed_mask[N,H,W])`.
    """
    n = state.n_robots
    height, width = state.height, state.width

    sensed_masks = np.zeros((n, height, width), dtype=bool)
    for i in range(n):
        cells = sensed_cells((int(state.robot_pos[i, 0]), int(state.robot_pos[i, 1])), sensor_radius, height, width)
        if len(cells) > 0:
            sensed_masks[i, cells[:, 0], cells[:, 1]] = True

    newly_discovered: list[int] = []

    for i in range(n):
        belief = state.beliefs[i]
        mask = sensed_masks[i]

        # Terrain: sensed cells become known with their true terrain value.
        belief.terrain_belief[mask] = state.terrain[mask]

        # Own visitation (own position is always sensed, radius >= 1).
        belief.visited_self[int(state.robot_pos[i, 0]), int(state.robot_pos[i, 1])] = True
        belief.visited_team_known |= belief.visited_self  # a robot always knows its own visits

        # Targets: any target whose cell is sensed becomes known to this robot.
        for k in range(state.n_targets):
            ty, tx = state.target_pos[k]
            if mask[ty, tx]:
                if not belief.target_known[k]:
                    belief.target_known[k] = True
                if state.target_status[k] == RESCUED:
                    belief.target_rescued_known[k] = True

        # Team-level knowledge (used by rewards/metrics; visible to actors
        # only under comm_mode == "shared_map", per §12.2).
        state.known_team |= mask

    # Team-level target discovery bookkeeping (independent of comm mode —
    # this drives the reward and the true `discovered`/`rescued` status,
    # which are world-truth, not per-robot belief).
    any_sensed = np.any(sensed_masks, axis=0)
    for k in range(state.n_targets):
        ty, tx = state.target_pos[k]
        if state.target_status[k] == HIDDEN and any_sensed[ty, tx]:
            state.target_status[k] = DISCOVERED
            state.target_t_discovered[k] = state.t
            newly_discovered.append(k)

    _apply_communication(state, comm_mode, comm_radius, sensed_masks)

    return newly_discovered, sensed_masks


def _apply_communication(
    state: WorldState, comm_mode: str, comm_radius: float, sensed_masks: np.ndarray
) -> None:
    n = state.n_robots
    if comm_mode == "none":
        return

    if comm_mode == "shared_map":
        groups = [list(range(n))]
    elif comm_mode == "limited_range":
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dy = state.robot_pos[i, 0] - state.robot_pos[j, 0]
                dx = state.robot_pos[i, 1] - state.robot_pos[j, 1]
                if dy * dy + dx * dx <= comm_radius * comm_radius:
                    edges.append((i, j))
        groups = _connected_components(n, edges)
    else:
        raise ValueError(f"unknown comm_mode: {comm_mode}")

    for group in groups:
        if len(group) < 2:
            continue
        _merge_group_beliefs(state, group)


def _merge_group_beliefs(state: WorldState, group: list[int]) -> None:
    """Merge (set-union) the beliefs of all robots in `group` (§19.3)."""
    beliefs = [state.beliefs[i] for i in group]

    merged_known = np.zeros_like(beliefs[0].terrain_belief, dtype=bool)
    for b in beliefs:
        merged_known |= b.known

    merged_target_known = np.zeros_like(beliefs[0].target_known)
    merged_target_rescued_known = np.zeros_like(beliefs[0].target_rescued_known)
    merged_visited_team_known = np.zeros_like(beliefs[0].visited_team_known)
    for b in beliefs:
        merged_target_known |= b.target_known
        merged_target_rescued_known |= b.target_rescued_known
        merged_visited_team_known |= b.visited_team_known

    for i in group:
        belief = state.beliefs[i]
        newly_known = merged_known & ~belief.known
        if np.any(newly_known):
            belief.terrain_belief[newly_known] = state.terrain[newly_known]
        belief.target_known |= merged_target_known
        belief.target_rescued_known |= merged_target_rescued_known
        belief.visited_team_known |= merged_visited_team_known

"""Greedy nearest-frontier reference heuristic (§14.2).

Non-learning, hand-coded, and used ONLY for: (a) `max_steps` calibration,
(b) a solvability check within the time limit, (c) a labelled context line
in comparison charts. It must never be presented as learned behaviour.

Per robot, INDEPENDENTLY in its planning (it only ever reads its own
belief — never another robot's belief, position history, or any
privileged world state) and respecting partial observability:
  1. If the belief contains a known, not-yet-rescued target, plan toward
     the nearest one (BFS through known-free cells only).
  2. Otherwise, plan toward the nearest frontier — a known-free cell
     adjacent to an unknown cell.
  3. If nothing is reachable through known space, WAIT.

Ties are broken deterministically by (BFS distance, cell) lexicographic
order (§14.2).

Destination-claim arbitration (a deliberate, documented deviation from
pure independence): two robots whose independent plans repeatedly select
the SAME destination cell trigger the environment's symmetric R2
conflict rule (§6.5) every step, with no way to escape, since neither
plan changes while their beliefs stay identical — a permanent deadlock
that would make this tool useless for its actual job (calibrating
`max_steps` and checking solvability, §14.2). To keep the heuristic
usable for that job without touching the environment's own
tie-breaking (which must stay symmetric — §6.5), each robot's ranked
list of candidate destinations is resolved SEQUENTIALLY by robot index
within one `act()` call: a robot whose top choice is already claimed by
an earlier robot this tick falls through to its next-best candidate.
This is bookkeeping internal to a hand-coded reference tool, not a
decentralized learned actor, so it does not touch the CTDE boundary —
each robot's underlying BFS still uses only that robot's own belief.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from marl.envs.core.types import (
    ACTION_DELTAS,
    DOWN,
    LEFT,
    N_ACTIONS,
    OBSTACLE,
    RIGHT,
    UP,
    WAIT,
    RobotBelief,
    WorldState,
)
from marl.envs.core.world import SwarmWorld

_NEIGHBOUR_ACTIONS = [(a, tuple(ACTION_DELTAS[a])) for a in range(1, N_ACTIONS)]  # skip WAIT

_REVERSE_ACTION = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT, WAIT: WAIT}


def _bfs_with_first_step(
    known_free: np.ndarray, start: tuple[int, int]
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """BFS over `known_free` from `start`. Returns (dist, first_action):
    for every reachable cell, the shortest-path distance from `start` and
    the ACTION taken at `start` that begins a shortest path there.
    """
    height, width = known_free.shape
    sy, sx = start
    dist: dict[tuple[int, int], int] = {start: 0}
    first_action: dict[tuple[int, int], int] = {start: WAIT}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        y, x = q.popleft()
        for action, (dy, dx) in _NEIGHBOUR_ACTIONS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if not known_free[ny, nx]:
                continue
            cell = (ny, nx)
            if cell in dist:
                continue
            dist[cell] = dist[(y, x)] + 1
            first_action[cell] = first_action[(y, x)] if (y, x) != start else action
            q.append(cell)

    return dist, first_action


def _ranked_candidates(
    state: WorldState, belief: RobotBelief, pos: tuple[int, int], height: int, width: int
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    """Return (first_action, ranked_destination_cells) for one robot: the
    nearest known unrescued target(s) first (if any), then frontiers, each
    group sorted by (distance, cell)."""
    known_free = belief.known & (belief.terrain_belief != OBSTACLE)
    known_free[pos] = True
    dist, first_action = _bfs_with_first_step(known_free, pos)

    target_candidates = []
    for k in range(state.n_targets):
        if belief.target_known[k] and not belief.target_rescued_known[k]:
            t = (int(state.target_pos[k, 0]), int(state.target_pos[k, 1]))
            if t in dist:
                target_candidates.append((dist[t], t))
    target_candidates.sort()

    frontier_candidates = []
    for (y, x), d in dist.items():
        for _, (dy, dx) in _NEIGHBOUR_ACTIONS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not belief.known[ny, nx]:
                frontier_candidates.append((d, (y, x)))
                break
    frontier_candidates.sort()

    ranked = [cell for _, cell in target_candidates] + [cell for _, cell in frontier_candidates]
    return first_action, ranked


class FrontierHeuristicPolicy:
    """Implements the `Policy` protocol."""

    def act(self, world: SwarmWorld, rng: np.random.Generator) -> np.ndarray:
        state = world.state
        if state is None:
            raise RuntimeError("FrontierHeuristicPolicy.act() called on a world before reset()")

        height, width = state.height, state.width
        actions = np.full(state.n_robots, WAIT, dtype=np.int32)
        # Seed with every robot's CURRENT position: the environment's R1
        # rule (§6.5) blocks a move into ANY cell occupied at the start of
        # the step, even if that occupant is also moving away this tick.
        # Without this, a robot processed early could "claim" a cell that
        # a not-yet-processed teammate still physically occupies, which
        # the environment would then reject anyway — producing exactly
        # the kind of avoidable, permanent standoff this arbitration
        # exists to prevent.
        occ0 = {(int(state.robot_pos[j, 0]), int(state.robot_pos[j, 1])) for j in range(state.n_robots)}
        claimed_destinations: set[tuple[int, int]] = set(occ0)

        for i in range(state.n_robots):
            belief = state.beliefs[i]
            pos = (int(state.robot_pos[i, 0]), int(state.robot_pos[i, 1]))
            first_action, ranked = _ranked_candidates(state, belief, pos, height, width)
            avoid_action = _REVERSE_ACTION.get(int(state.prev_action[i]), WAIT)

            chosen_action = WAIT
            chosen_dest = pos
            # Two passes: first, prefer a candidate that doesn't immediately
            # backtrack along the robot's last move (a common cause of
            # stable 2-cycle oscillations when the ranked list flips
            # between two equally-good candidates tick to tick, since
            # nothing in an unchanging local belief breaks that tie
            # otherwise). If nothing else is available, backtracking is
            # allowed — refusing it outright would deadlock a robot in a
            # genuine dead-end corridor where retreating is the only move.
            for avoid_backtrack in (True, False):
                for cell in ranked:
                    if cell == pos:
                        continue
                    a = first_action[cell]
                    if avoid_backtrack and a == avoid_action:
                        continue
                    dy, dx = ACTION_DELTAS[a]
                    dest = (pos[0] + dy, pos[1] + dx)
                    if dest in claimed_destinations:
                        continue
                    chosen_action = a
                    chosen_dest = dest
                    break
                if chosen_dest != pos:
                    break

            actions[i] = chosen_action
            # Claim this robot's resulting cell (whether it moves or stays)
            # so a LATER robot this tick doesn't plan into it.
            claimed_destinations.add(chosen_dest)

        return actions

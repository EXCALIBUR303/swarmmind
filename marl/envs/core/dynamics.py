"""Multi-agent movement dynamics: the no-follow conflict rule (§6.5).

`resolve_step` implements steps 2-5 of the step order in §6.4: propose
destinations, terrain check, robot-conflict resolution (single pass,
order-independent), and the resulting moved positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.envs.core.types import ACTION_DELTAS, OBSTACLE


@dataclass
class DynamicsResult:
    new_pos: np.ndarray  # int32[N, 2]
    moved: np.ndarray  # bool[N]
    blocked_by_robot: np.ndarray  # bool[N]
    bumped_wall: np.ndarray  # bool[N]


def resolve_step(
    positions: np.ndarray,  # int32[N, 2]
    actions: np.ndarray,  # int32[N], each in {0..4}
    terrain: np.ndarray,  # int8[H, W]
) -> DynamicsResult:
    """Resolve one joint step under the no-follow rule (§6.5).

    Precondition: every action is a valid index into ACTION_DELTAS
    (0..4). Validating that is the caller's job (§6.4 step 1) — this
    function assumes clean input, as the spec requires a ValueError for
    truly malformed actions to be raised earlier, not silently handled here.
    """
    n = positions.shape[0]
    height, width = terrain.shape

    occ0 = {(int(p[0]), int(p[1])) for p in positions}

    deltas = ACTION_DELTAS[actions]  # [N, 2]
    proposed = positions + deltas  # [N, 2]

    in_bounds = (
        (proposed[:, 0] >= 0) & (proposed[:, 0] < height) & (proposed[:, 1] >= 0) & (proposed[:, 1] < width)
    )
    # Clamp out-of-bounds proposals to current position for the terrain lookup,
    # then overwrite with "stay" below — this only avoids an index error.
    safe_proposed = np.where(in_bounds[:, None], proposed, positions)
    is_obstacle = terrain[safe_proposed[:, 0], safe_proposed[:, 1]] == OBSTACLE
    terrain_blocked = (~in_bounds) | is_obstacle
    bumped_wall = terrain_blocked & (~np.all(deltas == 0, axis=1))

    dest = np.where(terrain_blocked[:, None], positions, proposed)

    # Robot conflict resolution — single pass, order-independent (§6.5).
    blocked_by_robot = np.zeros(n, dtype=bool)

    is_moving = np.any(dest != positions, axis=1)

    # Rule R1: destination occupied at start of step (by ANY robot, moving or not).
    for i in range(n):
        if not is_moving[i]:
            continue
        dest_tuple = (int(dest[i, 0]), int(dest[i, 1]))
        if dest_tuple in occ0:
            blocked_by_robot[i] = True

    # Rule R2: contested empty cell — two or more (still-)moving robots share a destination.
    dest_counts: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        if not is_moving[i] or blocked_by_robot[i]:
            continue
        key = (int(dest[i, 0]), int(dest[i, 1]))
        dest_counts.setdefault(key, []).append(i)
    for indices in dest_counts.values():
        if len(indices) > 1:
            for i in indices:
                blocked_by_robot[i] = True

    final_pos = np.where(blocked_by_robot[:, None], positions, dest)
    moved = np.any(final_pos != positions, axis=1)

    return DynamicsResult(
        new_pos=final_pos.astype(np.int32),
        moved=moved,
        blocked_by_robot=blocked_by_robot,
        bumped_wall=bumped_wall,
    )

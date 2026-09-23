"""Actor observation construction: LOCAL ONLY (§8, §10.2).

This module must never read anything beyond a single robot's own belief,
its current sensing, and its own trajectory/state. It is the actor's only
window onto the world — see the CTDE information boundary (§12) and the
counterfactual leak tests (§12.4) that check this module's output is
invariant to hidden-target positions, unobserved terrain, and unsensed
teammates.
"""

from __future__ import annotations

import numpy as np

from marl.envs.core.types import ACTION_DELTAS, N_ACTIONS, OBSTACLE, WorldState

N_OBS_CHANNELS = 8
N_VEC_FEATURES = 10

# Channel indices (§8.1)
CH_OBSTACLE = 0
CH_KNOWN_FREE = 1
CH_UNKNOWN = 2
CH_SENSED_NOW = 3
CH_VISITED_SELF = 4
CH_VISITED_KNOWN = 5
CH_TEAMMATE = 6
CH_TARGET = 7


def build_actor_observation(
    state: WorldState,
    robot_idx: int,
    obs_radius: int,
    sensed_mask_i: np.ndarray,  # bool[H, W], this robot's sensing THIS step
    action_masking: bool,
    max_steps: int,
    prev_action: int,
    blocked_last_step: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (grid[8,K,K], vec[10], mask[5]) for one robot (§8, §10.2).

    K = 2*obs_radius + 1. The robot is always at the window's centre cell.
    Out-of-bounds window cells are treated as obstacle=1, known_free=0,
    unknown=0 (§8.1).
    """
    height, width = state.height, state.width
    belief = state.beliefs[robot_idx]
    cy, cx = int(state.robot_pos[robot_idx, 0]), int(state.robot_pos[robot_idx, 1])

    k = 2 * obs_radius + 1
    grid = np.zeros((N_OBS_CHANNELS, k, k), dtype=np.float32)

    # Window bounds in world coordinates.
    y0, y1 = cy - obs_radius, cy + obs_radius + 1
    x0, x1 = cx - obs_radius, cx + obs_radius + 1
    # Intersect with the actual grid.
    wy0, wy1 = max(y0, 0), min(y1, height)
    wx0, wx1 = max(x0, 0), min(x1, width)
    # Corresponding slice in the (k,k) window.
    ly0, ly1 = wy0 - y0, wy1 - y0
    lx0, lx1 = wx0 - x0, wx1 - x0

    # Default: out-of-bounds padding -> obstacle=1, unknown=0, known_free=0
    # (§8.1). The in-bounds region is then reset to "unknown" and filled
    # in from belief below.
    grid[CH_OBSTACLE, :, :] = 1.0
    grid[CH_OBSTACLE, ly0:ly1, lx0:lx1] = 0.0
    grid[CH_UNKNOWN, ly0:ly1, lx0:lx1] = 1.0

    if ly1 > ly0 and lx1 > lx0:
        sub_known = belief.known[wy0:wy1, wx0:wx1]
        sub_terrain = belief.terrain_belief[wy0:wy1, wx0:wx1]
        sub_obstacle = sub_known & (sub_terrain == OBSTACLE)
        sub_free = sub_known & (sub_terrain != OBSTACLE)

        grid[CH_OBSTACLE, ly0:ly1, lx0:lx1] = sub_obstacle.astype(np.float32)
        grid[CH_KNOWN_FREE, ly0:ly1, lx0:lx1] = sub_free.astype(np.float32)
        grid[CH_UNKNOWN, ly0:ly1, lx0:lx1] = (~sub_known).astype(np.float32)

        grid[CH_SENSED_NOW, ly0:ly1, lx0:lx1] = sensed_mask_i[wy0:wy1, wx0:wx1].astype(np.float32)
        grid[CH_VISITED_SELF, ly0:ly1, lx0:lx1] = belief.visited_self[wy0:wy1, wx0:wx1].astype(np.float32)
        grid[CH_VISITED_KNOWN, ly0:ly1, lx0:lx1] = belief.visited_team_known[wy0:wy1, wx0:wx1].astype(
            np.float32
        )

        # Teammates currently sensed (never remembered — §8.4).
        teammate_mask_sub = np.zeros((wy1 - wy0, wx1 - wx0), dtype=bool)
        for j in range(state.n_robots):
            if j == robot_idx:
                continue
            ty, tx = int(state.robot_pos[j, 0]), int(state.robot_pos[j, 1])
            if wy0 <= ty < wy1 and wx0 <= tx < wx1 and sensed_mask_i[ty, tx]:
                teammate_mask_sub[ty - wy0, tx - wx0] = True
        grid[CH_TEAMMATE, ly0:ly1, lx0:lx1] = teammate_mask_sub.astype(np.float32)

        # Known, not-known-rescued targets (§8.1 ch.7).
        target_mask_sub = np.zeros((wy1 - wy0, wx1 - wx0), dtype=bool)
        for kdx in range(state.n_targets):
            if not belief.target_known[kdx] or belief.target_rescued_known[kdx]:
                continue
            ty, tx = int(state.target_pos[kdx, 0]), int(state.target_pos[kdx, 1])
            if wy0 <= ty < wy1 and wx0 <= tx < wx1:
                target_mask_sub[ty - wy0, tx - wx0] = True
        grid[CH_TARGET, ly0:ly1, lx0:lx1] = target_mask_sub.astype(np.float32)

    # --- Vector features (§8.2) ---
    vec = np.zeros(N_VEC_FEATURES, dtype=np.float32)
    vec[0] = cx / max(width - 1, 1)
    vec[1] = cy / max(height - 1, 1)
    vec[2] = 1.0 - (state.t / max_steps)
    if prev_action >= 0:
        vec[3 + prev_action] = 1.0  # one-hot of previous action (indices 3..7)
    vec[8] = 1.0 if blocked_last_step else 0.0
    n_known_rescued = int(belief.target_rescued_known.sum())
    vec[9] = n_known_rescued / max(state.n_targets, 1)

    # --- Action mask (§10.2) ---
    mask = np.ones(N_ACTIONS, dtype=np.int8)
    if action_masking:
        for a in range(N_ACTIONS):
            dy, dx = ACTION_DELTAS[a]
            if dy == 0 and dx == 0:
                continue  # WAIT always valid
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < height and 0 <= nx < width):
                mask[a] = 0
                continue
            # Locality: this uses only current sensing (radius >= 1 always
            # covers the 4-neighbourhood), never remembered/global terrain.
            if sensed_mask_i[ny, nx] and state.terrain[ny, nx] == OBSTACLE:
                mask[a] = 0

    return grid, vec, mask

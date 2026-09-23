"""Multi-agent movement dynamics: the no-follow conflict rule (§6.5, §33.2)."""

from __future__ import annotations

import numpy as np
import pytest

from marl.envs.core.dynamics import resolve_step
from marl.envs.core.types import DOWN, FREE, LEFT, OBSTACLE, RIGHT, UP, WAIT


def _open_terrain(h: int, w: int) -> np.ndarray:
    return np.full((h, w), FREE, dtype=np.int8)


def test_move_into_free_uncontested_cell():
    terrain = _open_terrain(5, 5)
    pos = np.array([[2, 2]], dtype=np.int32)
    result = resolve_step(pos, np.array([RIGHT]), terrain)
    assert np.array_equal(result.new_pos, [[2, 3]])
    assert result.moved[0]
    assert not result.blocked_by_robot[0]
    assert not result.bumped_wall[0]


def test_wall_bump_out_of_bounds():
    terrain = _open_terrain(5, 5)
    pos = np.array([[0, 0]], dtype=np.int32)
    result = resolve_step(pos, np.array([UP]), terrain)
    assert np.array_equal(result.new_pos, [[0, 0]])
    assert not result.moved[0]
    assert result.bumped_wall[0]
    assert not result.blocked_by_robot[0]


def test_obstacle_bump():
    terrain = _open_terrain(3, 3)
    terrain[1, 2] = OBSTACLE
    pos = np.array([[1, 1]], dtype=np.int32)
    result = resolve_step(pos, np.array([RIGHT]), terrain)
    assert np.array_equal(result.new_pos, [[1, 1]])
    assert result.bumped_wall[0]
    assert not result.moved[0]


def test_wait_never_bumps_or_blocks():
    terrain = _open_terrain(3, 3)
    pos = np.array([[1, 1]], dtype=np.int32)
    result = resolve_step(pos, np.array([WAIT]), terrain)
    assert not result.moved[0]
    assert not result.bumped_wall[0]
    assert not result.blocked_by_robot[0]


def test_move_into_occupied_waiting_robot_blocks_mover_only():
    terrain = _open_terrain(3, 3)
    # Robot 0 at (1,1) tries to move right into robot 1 at (1,2), which waits.
    pos = np.array([[1, 1], [1, 2]], dtype=np.int32)
    result = resolve_step(pos, np.array([RIGHT, WAIT]), terrain)
    assert result.blocked_by_robot[0]
    assert not result.blocked_by_robot[1]
    assert np.array_equal(result.new_pos[0], [1, 1])
    assert np.array_equal(result.new_pos[1], [1, 2])


def test_swap_both_blocked():
    terrain = _open_terrain(3, 3)
    pos = np.array([[1, 1], [1, 2]], dtype=np.int32)
    result = resolve_step(pos, np.array([RIGHT, LEFT]), terrain)
    assert result.blocked_by_robot[0]
    assert result.blocked_by_robot[1]
    assert np.array_equal(result.new_pos, pos)
    assert not result.moved[0]
    assert not result.moved[1]


def test_follow_blocked():
    """A follows B into B's vacated cell: A must stay, B moves."""
    terrain = _open_terrain(3, 3)
    # A at (1,0) moves right to (1,1)=B's start. B at (1,1) moves right to (1,2).
    pos = np.array([[1, 0], [1, 1]], dtype=np.int32)
    result = resolve_step(pos, np.array([RIGHT, RIGHT]), terrain)
    assert result.blocked_by_robot[0]  # A (the follower) blocked
    assert not result.blocked_by_robot[1]  # B moves freely
    assert np.array_equal(result.new_pos[0], [1, 0])
    assert np.array_equal(result.new_pos[1], [1, 2])
    assert result.moved[1]
    assert not result.moved[0]


def test_contested_empty_cell_both_blocked():
    terrain = _open_terrain(3, 3)
    # A at (0,1) moves down to (1,1). B at (2,1) moves up to (1,1). Neither
    # was previously occupied there, so this is a pure R2 contest.
    pos = np.array([[0, 1], [2, 1]], dtype=np.int32)
    result = resolve_step(pos, np.array([DOWN, UP]), terrain)
    assert result.blocked_by_robot[0]
    assert result.blocked_by_robot[1]
    assert np.array_equal(result.new_pos, pos)


def test_four_cycle_rotation_all_blocked():
    """A ring of 4 robots, each moving into the next one's cell (a single
    4-connected step can't form a true 3-cycle on a grid, so this uses a
    4-cycle: A(0,0)->(0,1), B(0,1)->(1,1), C(1,1)->(1,0), D(1,0)->(0,0)).
    All four destinations are in occ0, so R1 blocks every robot."""
    terrain = _open_terrain(3, 3)
    pos = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int32)
    actions = np.array([RIGHT, DOWN, LEFT, UP])
    result = resolve_step(pos, actions, terrain)
    assert np.all(result.blocked_by_robot)
    assert np.array_equal(result.new_pos, pos)
    assert not np.any(result.moved)


def test_invalid_action_index_raises_indexerror_via_caller_contract():
    """resolve_step itself indexes ACTION_DELTAS[actions]; out-of-range
    values raise IndexError, matching the "programming error" contract —
    the caller (SwarmWorld.step) is responsible for the ValueError guard
    on out-of-range actions (§10.3), tested in test_world.py."""
    terrain = _open_terrain(3, 3)
    pos = np.array([[1, 1]], dtype=np.int32)
    with pytest.raises(IndexError):
        resolve_step(pos, np.array([99]), terrain)


def test_fuzz_no_overlaps_and_stay_on_free_cells():
    """10k random joint steps across 50 maps: positions always unique and
    on free cells (§33.2 fuzz requirement)."""
    rng = np.random.default_rng(123)
    n_robots = 5
    for _map_idx in range(50):
        h, w = 10, 10
        terrain = (rng.random((h, w)) < 0.15).astype(np.int8)
        free_ys, free_xs = np.where(terrain == FREE)
        if len(free_ys) < n_robots:
            continue
        idxs = rng.choice(len(free_ys), size=n_robots, replace=False)
        pos = np.stack([free_ys[idxs], free_xs[idxs]], axis=1).astype(np.int32)

        for _step in range(200):
            actions = rng.integers(0, 5, size=n_robots)
            result = resolve_step(pos, actions, terrain)
            pos = result.new_pos
            # Uniqueness
            unique_positions = {tuple(p) for p in pos}
            assert len(unique_positions) == n_robots, "robots must never overlap"
            # On free cells
            for y, x in pos:
                assert terrain[y, x] == FREE, "robots must never stand on an obstacle"
                assert 0 <= y < h and 0 <= x < w

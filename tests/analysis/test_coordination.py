"""Coordination metrics (§24) against hand-built position traces and
hand-built `EpisodeTrace`s with known answers (§33.4)."""

from __future__ import annotations

import numpy as np
import pytest

from marl.analysis.coordination import (
    compute_coordination_metrics,
    dispersion_speed,
    duplicate_exploration_ratio,
    inter_robot_distances,
    redundant_pursuit_and_allocation,
    sensed_masks_per_robot,
    visitation_overlap,
)
from marl.envs.core.types import EpisodeTrace


def test_der_no_overlap_is_zero():
    positions = [np.array([[0, 0], [9, 9]])]
    masks = sensed_masks_per_robot(positions, sensor_radius=1, height=10, width=10)
    assert duplicate_exploration_ratio(masks) == 0.0


def test_der_identical_positions_hits_the_n_minus_1_over_n_bound():
    # Both robots at the same cell: every sensed cell is shared, so
    # DER = 1 - |S|/(2|S|) = 0.5 = (N-1)/N for N=2 (§24.1's stated bound).
    positions = [np.array([[5, 5], [5, 5]])]
    masks = sensed_masks_per_robot(positions, sensor_radius=1, height=10, width=10)
    assert duplicate_exploration_ratio(masks) == pytest.approx(0.5)


def test_der_partial_overlap_matches_hand_calculation():
    # radius=1 discs are a 5-cell "plus". Adjacent robots at (3,3)/(3,4)
    # share exactly the two center cells: union=8, sum=10, DER=1-8/10=0.2.
    positions = [np.array([[3, 3], [3, 4]])]
    masks = sensed_masks_per_robot(positions, sensor_radius=1, height=10, width=10)
    assert duplicate_exploration_ratio(masks) == pytest.approx(0.2)


def test_der_cumulative_over_multiple_steps_and_up_to_truncation():
    # Step 0: robots overlap at (3,3)/(3,4) (5-cell "plus" discs sharing 2
    # cells: union 8, sum 10). Step 1: robots move to opposite corners
    # (0,0)/(9,9), each sensing 3 new, non-overlapping, boundary-clipped
    # cells. Cumulative: per-robot totals are now 5+3=8 each (sum 16); the
    # grand union is 8 (from step 0) + 3 + 3 (step 1, all new) = 14.
    # DER = 1 - 14/16 = 0.125 — lower than step 0 alone, because the new,
    # non-overlapping cells dilute the ratio even though the original
    # overlap is still counted in the union exactly once.
    positions = [
        np.array([[3, 3], [3, 4]]),
        np.array([[0, 0], [9, 9]]),
    ]
    full = sensed_masks_per_robot(positions, sensor_radius=1, height=10, width=10)
    assert duplicate_exploration_ratio(full) == pytest.approx(0.125)
    # up_to=0 restricts to step 0 only -> the original 0.2 (nothing has moved yet).
    truncated = sensed_masks_per_robot(positions, sensor_radius=1, height=10, width=10, up_to=0)
    assert duplicate_exploration_ratio(truncated) == pytest.approx(0.2)


def test_visitation_overlap_hand_calculation():
    positions = [
        np.array([[0, 0], [5, 5]]),
        np.array([[0, 1], [0, 1]]),  # both robots occupy (0,1) this step
        np.array([[0, 2], [5, 6]]),
    ]
    # visited: R0={(0,0),(0,1),(0,2)}, R1={(5,5),(0,1),(5,6)}
    # union=5 cells, (0,1) shared by both -> overlap = 1/5.
    assert visitation_overlap(positions) == pytest.approx(0.2)


def test_visitation_overlap_no_shared_cells_is_zero():
    positions = [np.array([[0, 0], [9, 9]]), np.array([[0, 1], [9, 8]])]
    assert visitation_overlap(positions) == 0.0


def test_inter_robot_distances_mean_and_min():
    positions = [
        np.array([[0, 0], [0, 3]]),  # dist 3
        np.array([[0, 0], [0, 4]]),  # dist 4
        np.array([[0, 0], [0, 5]]),  # dist 5
    ]
    mean, min_ = inter_robot_distances(positions)
    assert mean == pytest.approx(4.0)
    assert min_ == pytest.approx(3.0)


def test_inter_robot_distances_single_robot_is_zero():
    positions = [np.array([[0, 0]])]
    assert inter_robot_distances(positions) == (0.0, 0.0)


def test_dispersion_speed_finds_the_first_step_that_exceeds_sensor_radius():
    positions = [
        np.array([[0, 0], [0, 1]]),  # dist 1, <= r_s
        np.array([[0, 0], [0, 2]]),  # dist 2, == r_s, NOT strictly greater
        np.array([[0, 0], [0, 3]]),  # dist 3, > r_s=2 -> this is the answer
    ]
    assert dispersion_speed(positions, sensor_radius=2) == 2


def test_dispersion_speed_none_when_never_exceeded_or_single_robot():
    positions = [np.array([[0, 0], [0, 1]]), np.array([[0, 0], [0, 2]])]
    assert dispersion_speed(positions, sensor_radius=2) is None
    assert dispersion_speed([np.array([[0, 0]])], sensor_radius=2) is None


def _pursuit_trace(rescuer: int) -> EpisodeTrace:
    """2 robots, 1 target at (4,4) on an open 5x5 grid. Target discovered
    at step index 1 (positions[2]) and rescued at step index 2 (positions[3])
    by `rescuer`. Manhattan BFS distance on an open grid: |dy|+|dx|."""
    trace = EpisodeTrace(map_id="m", width=5, height=5, n_robots=2, n_targets=1)
    trace.positions = [
        np.array([[0, 0], [4, 0]]),  # t0: R0 d=6, R1 d=4 (not scored: before discovery)
        np.array([[1, 0], [4, 1]]),  # t1
        np.array([[2, 0], [4, 2]]),  # t2 = discovery: R0 d=6, R1 d=2
        np.array([[2, 1], [4, 3]]),  # t3 = rescue:   R0 d=5, R1 d=1
    ]
    trace.actions = [np.array([0, 0])] * 3
    trace.rewards = [np.zeros(2)] * 3
    trace.reward_components = [{}] * 3
    trace.moved = [np.array([True, True])] * 3
    trace.blocked_by_robot = [np.array([False, False])] * 3
    trace.bumped_wall = [np.array([False, False])] * 3
    trace.discovered_events = [[], [0], []]
    trace.rescued_events = [[], [], [(0, rescuer)]]
    trace.coverage_over_time = [0.1, 0.2, 0.3, 0.4]
    trace.success = True
    trace.final_t = 3
    return trace


def _open_free(height: int = 5, width: int = 5) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def test_redundant_pursuit_and_allocation_when_the_nearest_robot_rescues():
    trace = _pursuit_trace(rescuer=1)
    pursuit, allocation = redundant_pursuit_and_allocation(trace, _open_free(), targets=[(4, 4)])
    # Both robots' distance to (4,4) strictly decreases from t2->t3
    # (R0: 6->5, R1: 2->1), over the single pursuit step -> mean = 2.0.
    assert pursuit == pytest.approx(2.0)
    # At discovery (t2) R1 (d=2) is nearer than R0 (d=6), and R1 rescues.
    assert allocation == pytest.approx(1.0)


def test_redundant_pursuit_and_allocation_when_the_farther_robot_rescues():
    trace = _pursuit_trace(rescuer=0)
    _pursuit, allocation = redundant_pursuit_and_allocation(trace, _open_free(), targets=[(4, 4)])
    # R0 (d=6 at discovery) is farther than R1 (d=2), but R0 is credited
    # with the rescue -> not the BFS-nearest robot.
    assert allocation == pytest.approx(0.0)


def test_redundant_pursuit_and_allocation_none_when_nothing_rescued():
    trace = _pursuit_trace(rescuer=1)
    trace.rescued_events = [[], [], []]
    pursuit, allocation = redundant_pursuit_and_allocation(trace, _open_free(), targets=[(4, 4)])
    assert pursuit is None and allocation is None


def test_compute_coordination_metrics_integration():
    trace = _pursuit_trace(rescuer=1)
    terrain = np.zeros((5, 5), dtype=np.int8)  # all FREE (0)
    m = compute_coordination_metrics(trace, terrain, targets=[(4, 4)], sensor_radius=1)
    assert m.redundant_pursuit == pytest.approx(2.0)
    assert m.allocation_optimality == pytest.approx(1.0)
    assert m.mean_inter_robot_distance > 0
    assert m.der >= 0.0
    # Coverage never reaches 0.5 in this trace (max 0.4) -> DER@50 undefined.
    assert m.der_at_50 is None

"""Episode metrics tests against hand-crafted traces (§23, §33.4)."""

from __future__ import annotations

import numpy as np

from marl.analysis.metrics import _gini, compute_episode_metrics
from marl.envs.core.types import EpisodeTrace


def _make_trace(n_robots=2, n_targets=1) -> EpisodeTrace:
    """A hand-built 3-step trace with known answers:
    - t=0: robot 0 moves right (explores 1 new cell), robot 1 waits.
    - t=1: robot 0 moves right again (discovers the target), robot 1 waits.
    - t=2: robot 0 moves right onto the target (rescues it, success).
    """
    trace = EpisodeTrace(map_id="m", width=10, height=10, n_robots=n_robots, n_targets=n_targets)
    trace.positions = [
        np.array([[0, 0], [5, 5]]),
        np.array([[0, 1], [5, 5]]),
        np.array([[0, 2], [5, 5]]),
        np.array([[0, 3], [5, 5]]),
    ]
    trace.actions = [np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
    trace.rewards = [
        np.array([0.1, 0.1]),
        np.array([0.1 + 0.25, 0.1 + 0.25]),
        np.array([2.0 + 0.1, 2.0 + 0.1]),
    ]
    trace.reward_components = [
        {"explore": np.array([0.1, 0.1]), "discover": np.array([0.0, 0.0]), "rescue": np.array([0.0, 0.0]), "complete": np.array([0.0, 0.0]), "time": np.array([0.01, 0.01]), "collision": np.array([0.0, 0.0])},
        {"explore": np.array([0.1, 0.1]), "discover": np.array([0.25, 0.25]), "rescue": np.array([0.0, 0.0]), "complete": np.array([0.0, 0.0]), "time": np.array([0.01, 0.01]), "collision": np.array([0.0, 0.0])},
        {"explore": np.array([0.0, 0.0]), "discover": np.array([0.0, 0.0]), "rescue": np.array([1.0, 1.0]), "complete": np.array([2.0, 2.0]), "time": np.array([0.01, 0.01]), "collision": np.array([0.0, 0.0])},
    ]
    trace.moved = [np.array([True, False]), np.array([True, False]), np.array([True, False])]
    trace.blocked_by_robot = [np.array([False, False])] * 3
    trace.bumped_wall = [np.array([False, False])] * 3
    trace.discovered_events = [[], [0], []]
    trace.rescued_events = [[], [], [(0, 0)]]
    trace.coverage_over_time = [0.1, 0.2, 0.3, 0.4]
    trace.explore_credit = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0])]
    trace.success = True
    trace.final_t = 3
    return trace


def test_basic_counts():
    trace = _make_trace()
    m = compute_episode_metrics(trace, n_reachable_free=10)
    assert m.success is True
    assert m.n_discovered == 1
    assert m.n_rescued == 1
    assert m.discover_rate == 1.0
    assert m.rescue_rate == 1.0
    assert m.steps_to_completion == 3


def test_distance_and_idle_fraction():
    trace = _make_trace()
    m = compute_episode_metrics(trace, n_reachable_free=10)
    assert m.distance_travelled_per_robot == [3.0, 0.0]
    assert m.distance_travelled_total == 3.0
    assert m.idle_fraction_per_robot == [0.0, 1.0]


def test_collisions_and_invalid_actions_zero():
    trace = _make_trace()
    m = compute_episode_metrics(trace, n_reachable_free=10)
    assert m.n_collisions == 0
    assert m.n_invalid_actions == 0


def test_rescue_latency():
    trace = _make_trace()
    m = compute_episode_metrics(trace, n_reachable_free=10)
    # discovered at t=1 (index into discovered_events), rescued at t=2 -> latency 1
    assert m.rescue_latency_mean == 1.0


def test_rescue_latency_none_when_nothing_rescued():
    trace = _make_trace()
    trace.rescued_events = [[], [], []]
    trace.success = False
    m = compute_episode_metrics(trace, n_reachable_free=10)
    assert m.rescue_latency_mean is None


def test_oscillation_rate_detects_back_and_forth():
    trace = EpisodeTrace(map_id="m", width=10, height=10, n_robots=1, n_targets=1)
    # positions: (0,0) -> (0,1) -> (0,0) -> (0,1): a pure back-and-forth.
    trace.positions = [
        np.array([[0, 0]]),
        np.array([[0, 1]]),
        np.array([[0, 0]]),
        np.array([[0, 1]]),
    ]
    trace.actions = [np.array([4]), np.array([3]), np.array([4])]
    trace.rewards = [np.array([0.0])] * 3
    trace.reward_components = [{"explore": np.array([0.0])}] * 3
    trace.moved = [np.array([True])] * 3
    trace.blocked_by_robot = [np.array([False])] * 3
    trace.bumped_wall = [np.array([False])] * 3
    trace.discovered_events = [[], [], []]
    trace.rescued_events = [[], [], []]
    trace.coverage_over_time = [0.1, 0.1, 0.1, 0.1]
    trace.explore_credit = [np.zeros(1)] * 3
    trace.success = False
    trace.final_t = 3

    m = compute_episode_metrics(trace, n_reachable_free=10)
    # Every move after t=0 is an oscillation (moves=2 counted at t=1,2; both flagged)
    assert m.oscillation_rate_per_robot[0] > 0.0


def test_workload_gini_zero_when_perfectly_equal():
    shares = np.array([0.5, 0.5])
    assert abs(_gini(shares)) < 1e-9


def test_workload_gini_positive_when_unequal():
    shares = np.array([0.9, 0.1])
    assert _gini(shares) > 0.0


def test_workload_gini_handles_empty():
    assert _gini(np.array([])) == 0.0


def test_zero_step_trace_does_not_crash():
    trace = EpisodeTrace(map_id="m", width=10, height=10, n_robots=1, n_targets=1)
    trace.positions = [np.array([[0, 0]])]
    trace.coverage_over_time = [0.1]
    trace.success = False
    trace.final_t = 0
    m = compute_episode_metrics(trace, n_reachable_free=10)
    assert m.n_discovered == 0
    assert m.steps_to_completion == 0


def test_target_sensed_at_reset_counts_as_discovered():
    """Phase 15 regression: a target already sensed by the reset-time sensing pass appears in no
    step's `discovered_events`; it used to be missing from n_discovered (discover_rate < rescue_rate)."""
    trace = _make_trace()
    trace.discovered_events = [[], [], []]
    trace.initial_discovered = [0]
    m = compute_episode_metrics(trace, n_reachable_free=10)
    assert m.n_discovered == 1 and m.discover_rate >= m.rescue_rate
    assert m.rescue_latency_mean == 2  # discovered at reset (t=0), rescued during step index 2


def test_workload_share_uses_observer_credit_not_movement():
    """Phase 15 regression (§23): credit follows who sensed the newly known cells. A robot that moves
    but reveals nothing earns nothing; the terminal payout (in the explore reward) earns nothing."""
    trace = _make_trace()
    m = compute_episode_metrics(trace, n_reachable_free=10)
    assert m.workload_share_per_robot == [1.0, 0.0]
    trace.explore_credit = [np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([0.0, 0.0])]
    m = compute_episode_metrics(trace, n_reachable_free=10)
    assert m.workload_share_per_robot == [0.5, 0.5] and abs(m.workload_gini) < 1e-12

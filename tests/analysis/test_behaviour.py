"""Behaviour analysis (§26) against hand-built traces (§33.4)."""

from __future__ import annotations

import numpy as np

from marl.analysis.behaviour import (
    accumulate_heatmaps,
    der_over_time,
    discovery_timeline,
    episode_behaviour,
    inter_robot_distance_over_time,
    obstacle_mask,
)
from marl.envs.core.types import OBSTACLE, EpisodeTrace


def _trace() -> EpisodeTrace:
    trace = EpisodeTrace(map_id="m1", width=5, height=5, n_robots=2, n_targets=1)
    trace.positions = [
        np.array([[0, 0], [0, 4]]),
        np.array([[0, 1], [0, 3]]),
        np.array([[0, 2], [0, 2]]),  # they meet at the same cell
    ]
    trace.actions = [np.array([4, 3])] * 2
    trace.rewards = [np.zeros(2)] * 2
    trace.reward_components = [{"explore": np.array([0.1, 0.1])}] * 2
    trace.explore_credit = [np.array([3.0, 3.0]), np.array([1.0, 1.0])]
    trace.moved = [np.array([True, True])] * 2
    trace.blocked_by_robot = [np.array([False, False]), np.array([True, False])]
    trace.bumped_wall = [np.array([False, False])] * 2
    trace.discovered_events = [[], [0]]
    trace.rescued_events = [[], []]
    trace.coverage_over_time = [0.1, 0.3, 0.5]
    trace.success = False
    trace.final_t = 2
    return trace


def test_der_over_time_length_and_monotonic_union_growth():
    trace = _trace()
    curve = der_over_time(trace.positions, sensor_radius=1, height=5, width=5)
    assert len(curve) == len(trace.positions)
    assert all(0.0 <= v <= 1.0 for v in curve)
    # Robots converge onto the same cell by the last step -> maximal overlap
    # for N=2 discs of equal size once fully coincident.
    assert curve[-1] > curve[0]


def test_inter_robot_distance_over_time_hand_calc():
    trace = _trace()
    curve = inter_robot_distance_over_time(trace.positions)
    assert len(curve) == 3
    assert curve[0]["mean"] == curve[0]["min"] == 4.0  # (0,0)-(0,4)
    assert curve[1]["mean"] == curve[1]["min"] == 2.0  # (0,1)-(0,3)
    assert curve[2]["mean"] == curve[2]["min"] == 0.0  # same cell


def test_discovery_timeline_marks_undiscovered_targets_as_none():
    trace = _trace()
    timeline = discovery_timeline(trace)
    assert timeline == [{"target": 0, "t_discovered": 2, "t_rescued": None, "rescued_by": None}]


def test_discovery_timeline_records_rescue():
    trace = _trace()
    trace.rescued_events = [[], [(0, 1)]]
    timeline = discovery_timeline(trace)
    assert timeline == [{"target": 0, "t_discovered": 2, "t_rescued": 2, "rescued_by": 1}]


def test_episode_behaviour_bundle_has_expected_keys_and_lengths():
    trace = _trace()
    bundle = episode_behaviour(trace, n_reachable_free=20, sensor_radius=1)
    assert bundle["map_id"] == "m1"
    assert len(bundle["coverage_over_time"]) == 3
    assert len(bundle["der_over_time"]) == 3
    assert len(bundle["inter_robot_distance_over_time"]) == 3
    assert len(bundle["workload_share_per_robot"]) == 2
    assert len(bundle["idle_fraction_per_robot"]) == 2


def test_accumulate_heatmaps_counts_visits_and_collisions():
    t1 = _trace()
    t2 = _trace()
    heat = accumulate_heatmaps([t1, t2], height=5, width=5)
    visit = np.array(heat["visit_counts"])
    collide = np.array(heat["collision_counts"])
    assert visit.shape == (5, 5)
    # (0,2) is visited by both robots at the final step, across 2 episodes -> 4.
    assert visit[0, 2] == 4
    # blocked_by_robot marks robot 0 as blocked at step index 1 -> position
    # after that step is positions[2] = (0,2), across 2 episodes -> 2.
    assert collide[0, 2] == 2
    assert collide.sum() == 2


def test_obstacle_mask_matches_terrain():
    terrain = np.array([[0, 1], [1, 0]], dtype=np.int8)
    assert obstacle_mask(terrain) == [[terrain[0, 0] == OBSTACLE, True], [True, terrain[1, 1] == OBSTACLE]]

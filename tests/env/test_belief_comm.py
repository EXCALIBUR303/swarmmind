"""Communication mode tests: none / shared_map / limited_range (§19.3)."""

from __future__ import annotations

from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.world import SwarmWorld


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t",
        width=14,
        height=14,
        n_robots=3,
        n_targets=1,
        obstacle_density=0.0,
        max_steps=100,
        sensor_radius=1.0,
        obs_radius=3,
        target_min_dist=1,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def _empty_map(width, height, starts, targets) -> MapSpec:
    rows = ["." * width for _ in range(height)]
    return MapSpec(id="m", width=width, height=height, terrain_rows=rows, robot_starts=starts, targets=targets)


def test_mode_shared_map_discovery_appears_in_all_beliefs_same_step():
    cfg = _cfg(n_robots=2, n_targets=1, comm_mode="shared_map")
    starts = [(1, 1), (10, 10)]
    targets = [(10, 11)]  # right next to robot 1 only
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)  # robot 1's initial sense discovers it; shared_map broadcasts to robot 0 too

    assert world.state.beliefs[0].target_known[0]
    assert world.state.beliefs[1].target_known[0]


def test_mode_none_isolates_beliefs():
    cfg = _cfg(n_robots=2, n_targets=1, comm_mode="none")
    starts = [(1, 1), (10, 10)]
    targets = [(10, 11)]
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    assert not world.state.beliefs[0].target_known[0]
    assert world.state.beliefs[1].target_known[0]


def test_mode_limited_range_merge_within_radius():
    cfg = _cfg(n_robots=2, n_targets=1, comm_mode="limited_range", comm_radius=3.0)
    starts = [(5, 5), (5, 6)]  # within radius 3
    targets = [(5, 7)]  # sensed by robot 1 only (adjacent)
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    assert world.state.beliefs[0].target_known[0]  # merged from robot 1
    assert world.state.beliefs[1].target_known[0]


def test_mode_limited_range_out_of_range_isolated():
    cfg = _cfg(n_robots=2, n_targets=1, comm_mode="limited_range", comm_radius=2.0)
    starts = [(1, 1), (10, 10)]  # far apart, out of comm_radius
    targets = [(10, 11)]
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    assert not world.state.beliefs[0].target_known[0]
    assert world.state.beliefs[1].target_known[0]


def test_mode_limited_range_transitive_chain():
    """A-B-C chain, each within range of the next but A and C out of each
    other's direct range: knowledge must still propagate transitively."""
    cfg = _cfg(n_robots=3, n_targets=1, comm_mode="limited_range", comm_radius=3.0)
    starts = [(1, 1), (1, 4), (1, 7)]  # A-B dist 3, B-C dist 3, A-C dist 6 (out of range directly)
    targets = [(1, 8)]  # sensed by C only
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    assert world.state.beliefs[0].target_known[0]  # A learns via B
    assert world.state.beliefs[1].target_known[0]
    assert world.state.beliefs[2].target_known[0]


def test_visited_team_known_merges_under_shared_map():
    cfg = _cfg(n_robots=2, n_targets=1, comm_mode="shared_map")
    starts = [(1, 1), (10, 10)]
    targets = [(13, 13)]
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    assert world.state.beliefs[0].visited_team_known[10, 10]
    assert world.state.beliefs[1].visited_team_known[1, 1]


def test_visited_team_known_isolated_under_none():
    cfg = _cfg(n_robots=2, n_targets=1, comm_mode="none")
    starts = [(1, 1), (10, 10)]
    targets = [(13, 13)]
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    assert not world.state.beliefs[0].visited_team_known[10, 10]
    assert world.state.beliefs[0].visited_team_known[1, 1]  # own visit always known

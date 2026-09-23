"""Reward function tests: invariants and anti-hacking safeguards (§13, §33.2)."""

from __future__ import annotations

import numpy as np

from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.world import SwarmWorld


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t",
        width=6,
        height=6,
        n_robots=1,
        n_targets=1,
        obstacle_density=0.0,
        max_steps=50,
        sensor_radius=1.0,
        obs_radius=3,
        target_min_dist=1,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def _empty_map(width, height, starts, targets) -> MapSpec:
    rows = ["." * width for _ in range(height)]
    return MapSpec(id="m", width=width, height=height, terrain_rows=rows, robot_starts=starts, targets=targets)


def test_exploration_budget_invariant_on_success():
    """Total exploration-component reward over a successful episode must
    equal w_x * (|G| - |K_0|) exactly (§13.3)."""
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, sensor_radius=1.0)
    starts = [(2, 2)]
    targets = [(2, 3)]  # one step away
    m = _empty_map(6, 6, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    n_free = int(np.sum(world.state.terrain != 1))
    k0 = int(np.sum(world.state.known_team))
    w_x = cfg.reward.explore_total / n_free
    expected_total_explore = w_x * (n_free - k0)

    from marl.envs.core.types import RIGHT

    total_explore = 0.0
    result = world.step(np.array([RIGHT]))
    total_explore += float(result.reward_components["explore"][0])
    assert result.success

    assert abs(total_explore - expected_total_explore) < 1e-5


def test_exploration_budget_invariant_multi_step():
    """Same invariant, but over a longer episode with several exploration
    steps before the final rescue."""
    cfg = _cfg(width=8, height=8, n_robots=1, n_targets=1, sensor_radius=1.0, max_steps=50)
    starts = [(0, 0)]
    targets = [(0, 5)]
    m = _empty_map(8, 8, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    n_free = int(np.sum(world.state.terrain != 1))
    k0 = int(np.sum(world.state.known_team))
    w_x = cfg.reward.explore_total / n_free
    expected_total_explore = w_x * (n_free - k0)

    from marl.envs.core.types import RIGHT

    total_explore = 0.0
    for _ in range(5):
        result = world.step(np.array([RIGHT]))
        total_explore += float(result.reward_components["explore"][0])
        if result.success:
            break

    assert result.success
    assert abs(total_explore - expected_total_explore) < 1e-5


def test_no_double_counting_on_simultaneous_discovery():
    """Two robots that simultaneously bring the same NEW target within
    sensing range must not double-count the discover reward."""
    cfg = _cfg(width=8, height=8, n_robots=2, n_targets=1, sensor_radius=2.0, comm_mode="shared_map")
    starts = [(0, 0), (7, 7)]
    targets = [(3, 3)]
    m = _empty_map(8, 8, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    from marl.envs.core.types import RIGHT

    # Move both robots toward the target over several steps; whenever the
    # target becomes newly discovered, discover reward must be credited
    # exactly once (cfg.reward.discover), never doubled.
    discover_events = 0
    for _ in range(10):
        result = world.step(np.array([RIGHT, RIGHT]))
        if len(result.discovered) > 0:
            discover_events += 1
            assert abs(float(result.reward_components["discover"][0]) - cfg.reward.discover) < 1e-6
        if len(result.discovered) > 1:
            raise AssertionError("target discovered twice in the same step")
    assert discover_events <= 1  # a single target can only be discovered once, ever


def test_team_components_identical_across_agents():
    """Shared components (explore, discover, rescue, complete) must be
    identical across all robots each step; only collision/time may differ
    in principle (time is constant; collision is individual)."""
    cfg = _cfg(width=8, height=8, n_robots=2, n_targets=1, sensor_radius=1.0, comm_mode="shared_map")
    starts = [(0, 0), (7, 7)]
    targets = [(3, 3)]
    m = _empty_map(8, 8, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    from marl.envs.core.types import RIGHT

    result = world.step(np.array([RIGHT, RIGHT]))
    for key in ("explore", "discover", "rescue", "complete"):
        assert result.reward_components[key][0] == result.reward_components[key][1]


def test_collision_penalty_only_on_blocked_movers_not_waiters():
    cfg = _cfg(width=6, height=6, n_robots=2, n_targets=1, sensor_radius=1.0)
    starts = [(1, 1), (1, 2)]
    targets = [(4, 4)]
    m = _empty_map(6, 6, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    from marl.envs.core.types import RIGHT, WAIT

    result = world.step(np.array([RIGHT, WAIT]))  # robot 0 blocked moving into robot 1 (waiting)
    assert result.blocked_by_robot[0]
    assert not result.blocked_by_robot[1]
    assert abs(float(result.reward_components["collision"][0]) - cfg.reward.collision) < 1e-6
    assert result.reward_components["collision"][1] == 0.0


def test_time_cost_applied_every_step_regardless_of_action():
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, sensor_radius=1.0)
    starts = [(1, 1)]
    targets = [(4, 4)]
    m = _empty_map(6, 6, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    from marl.envs.core.types import WAIT

    result = world.step(np.array([WAIT]))
    assert abs(float(result.reward_components["time"][0]) - cfg.reward.time_cost) < 1e-6


def test_rescue_and_complete_reward_on_final_target():
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, sensor_radius=1.0)
    starts = [(1, 1)]
    targets = [(1, 2)]
    m = _empty_map(6, 6, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    from marl.envs.core.types import RIGHT

    result = world.step(np.array([RIGHT]))
    assert result.success
    assert len(result.rescued) == 1
    assert abs(float(result.reward_components["rescue"][0]) - cfg.reward.rescue) < 1e-6
    assert float(result.reward_components["complete"][0]) >= cfg.reward.complete - 1e-6

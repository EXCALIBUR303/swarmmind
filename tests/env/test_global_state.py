"""Global state (critic-only) tests (§9, §12.4 reverse-direction sanity check)."""

from __future__ import annotations

import numpy as np

from marl.envs.core.config import EnvConfig
from marl.envs.core.global_state import (
    CH_TARGETS_ALL,
    N_GLOBAL_CHANNELS,
    N_GLOBAL_VEC_FEATURES,
)
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.world import SwarmWorld


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t",
        width=10,
        height=10,
        n_robots=2,
        n_targets=2,
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


def test_global_state_shapes():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    world.reset(seed=1)
    shared_grid, vec = world.global_state()
    assert shared_grid.shape == (7, 10, 10)
    assert vec.shape == (N_GLOBAL_VEC_FEATURES,)
    full = world.full_global_grid(0)
    assert full.shape == (N_GLOBAL_CHANNELS, 10, 10)


def test_global_state_sees_hidden_target_but_actor_does_not():
    """The critic sanity check (§12.4 reverse direction): the global state
    MUST include hidden targets, proving the privileged channel actually
    carries the information the actor is denied."""
    cfg = _cfg(width=14, height=14, n_robots=1, n_targets=1, sensor_radius=1.0, comm_mode="none")
    starts = [(1, 1)]
    targets = [(10, 10)]  # far from robot, never sensed
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)

    assert world.state.target_status[0] == 0  # HIDDEN (world truth)

    shared_grid, _ = world.global_state()
    assert shared_grid[CH_TARGETS_ALL, 10, 10] == 1.0  # critic sees it

    from marl.envs.core.observations import CH_TARGET

    g_actor, _, _ = world.actor_obs(0)
    assert np.sum(g_actor[CH_TARGET]) == 0.0  # actor does not


def test_global_state_updates_with_robot_occupancy():
    cfg = _cfg(width=8, height=8, n_robots=2, n_targets=1, sensor_radius=1.0)
    starts = [(1, 1), (5, 5)]
    targets = [(7, 7)]
    m = _empty_map(8, 8, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    from marl.envs.core.global_state import CH_ROBOT_OCCUPANCY

    shared_grid, _ = world.global_state()
    assert shared_grid[CH_ROBOT_OCCUPANCY, 1, 1] == 1.0
    assert shared_grid[CH_ROBOT_OCCUPANCY, 5, 5] == 1.0
    assert np.sum(shared_grid[CH_ROBOT_OCCUPANCY]) == 2.0


def test_agent_specific_channels_differ_between_robots():
    cfg = _cfg(width=8, height=8, n_robots=2, n_targets=1, sensor_radius=1.0)
    starts = [(1, 1), (5, 5)]
    targets = [(7, 7)]
    m = _empty_map(8, 8, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    a0 = world.agent_state_channels(0)
    a1 = world.agent_state_channels(1)
    assert not np.array_equal(a0[0], a1[0])  # position channel differs


def test_coverage_vec_matches_known_team_fraction():
    cfg = _cfg(width=10, height=10, n_robots=1, n_targets=1, sensor_radius=1.0, obstacle_density=0.0)
    world = SwarmWorld(cfg)
    world.reset(seed=5)
    _, vec = world.global_state()
    known_free = np.sum(world.state.known_team & (world.state.terrain != 1))
    n_free = np.sum(world.state.terrain != 1)
    assert abs(vec[2] - known_free / n_free) < 1e-6

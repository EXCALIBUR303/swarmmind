"""Actor observation correctness, mask locality, and the CTDE information
leak tests (§8, §10.2, §12.4). These are the most important tests in the
whole environment: they prove the actor never sees privileged information.
"""

from __future__ import annotations

import numpy as np

from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.observations import (
    CH_KNOWN_FREE,
    CH_OBSTACLE,
    CH_SENSED_NOW,
    CH_TARGET,
    CH_TEAMMATE,
    CH_UNKNOWN,
    CH_VISITED_SELF,
    N_OBS_CHANNELS,
    N_VEC_FEATURES,
)
from marl.envs.core.types import N_ACTIONS, WAIT
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
        comm_mode="none",
        target_min_dist=1,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def _empty_map(width: int, height: int, starts, targets, **kwargs) -> MapSpec:
    rows = ["." * width for _ in range(height)]
    return MapSpec(
        id="handmade",
        width=width,
        height=height,
        terrain_rows=rows,
        robot_starts=starts,
        targets=targets,
    )


def test_obs_shapes_and_dtypes():
    cfg = _cfg(obs_radius=5, sensor_radius=2.0)
    world = SwarmWorld(cfg)
    world.reset(seed=1)
    g, v, m = world.actor_obs(0)
    k = 2 * 5 + 1
    assert g.shape == (N_OBS_CHANNELS, k, k)
    assert g.dtype == np.float32
    assert v.shape == (N_VEC_FEATURES,)
    assert v.dtype == np.float32
    assert m.shape == (N_ACTIONS,)
    assert np.all((g >= 0) & (g <= 1))
    assert np.all((v >= 0) & (v <= 1))


def test_channels_0_1_2_partition_in_bounds_cells():
    """obstacle/known_free/unknown must be a one-hot partition over
    in-bounds cells (§8.1)."""
    cfg = _cfg(width=8, height=8, obs_radius=4, sensor_radius=2.0)
    world = SwarmWorld(cfg)
    world.reset(seed=2)
    g, _, _ = world.actor_obs(0)
    total = g[CH_OBSTACLE] + g[CH_KNOWN_FREE] + g[CH_UNKNOWN]
    # Out-of-bounds cells get obstacle=1, known_free=0, unknown=0 -> sum=1.
    # In-bounds cells: exactly one of the three is 1 -> sum=1.
    assert np.allclose(total, 1.0)


def test_self_at_window_centre():
    cfg = _cfg(obs_radius=5)
    world = SwarmWorld(cfg)
    world.reset(seed=3)
    g, _, _ = world.actor_obs(0)
    center = 5
    # The robot's own cell is always sensed (radius >= 1), so it must show
    # up as sensed_now at the centre, and never as unknown.
    assert g[CH_SENSED_NOW, center, center] == 1.0
    assert g[CH_UNKNOWN, center, center] == 0.0


def test_out_of_bounds_padding():
    """A robot near a corner: the observation window extends past the grid
    edge. Padding must read obstacle=1, unknown=0, known_free=0 there."""
    cfg = _cfg(width=6, height=6, obs_radius=5, sensor_radius=1.0)
    starts = [(0, 0), (5, 5)]
    targets = [(2, 2), (3, 3)]
    m = _empty_map(6, 6, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    g, _, _ = world.actor_obs(0)  # robot at (0,0), window radius 5 -> mostly out of bounds
    # Top-left corner of the window (far out of bounds) should be padding.
    assert g[CH_OBSTACLE, 0, 0] == 1.0
    assert g[CH_UNKNOWN, 0, 0] == 0.0
    assert g[CH_KNOWN_FREE, 0, 0] == 0.0


def test_visited_self_channel_updates_after_move():
    cfg = _cfg(width=8, height=8, obs_radius=4, sensor_radius=1.0)
    starts = [(1, 1), (6, 6)]
    targets = [(4, 4), (5, 5)]
    m = _empty_map(8, 8, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    from marl.envs.core.types import RIGHT

    world.step(np.array([RIGHT, WAIT]))
    g, _, _ = world.actor_obs(0)  # robot 0 now at (1,2)
    center = 4
    assert g[CH_VISITED_SELF, center, center] == 1.0  # current cell


def test_teammate_channel_only_when_currently_sensed():
    """Teammates are NEVER remembered — only visible while within sensor
    range this step (§8.1 ch.6, §8.4)."""
    cfg = _cfg(width=10, height=10, obs_radius=5, sensor_radius=1.0)
    # Robot 0 and robot 1 adjacent -> sensed. Robot 2 far away -> not sensed.
    starts = [(5, 5), (5, 6), (0, 0)]
    targets = [(9, 9)]
    m = _empty_map(10, 10, starts, targets)
    cfg = _cfg(width=10, height=10, obs_radius=5, sensor_radius=1.0, n_robots=3, n_targets=1)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    g0, _, _ = world.actor_obs(0)
    assert np.sum(g0[CH_TEAMMATE]) == 1.0  # only robot 1, adjacent and sensed

    from marl.envs.core.types import RIGHT, UP

    # Move robot 1 away so it's no longer sensed by robot 0.
    world.step(np.array([UP, RIGHT, WAIT]))
    world.step(np.array([UP, RIGHT, WAIT]))
    world.step(np.array([UP, RIGHT, WAIT]))
    g0b, _, _ = world.actor_obs(0)
    assert np.sum(g0b[CH_TEAMMATE]) == 0.0


# ---------------------------------------------------------------------
# §12.4 Counterfactual leak tests — THE critical CTDE boundary tests.
# ---------------------------------------------------------------------


def test_counterfactual_hidden_target_invisible_to_actor():
    """Two world states differing ONLY in a HIDDEN target's position
    (outside all robots' sensing) must produce bit-identical actor
    observations and masks for every robot."""
    cfg = _cfg(width=12, height=12, obs_radius=5, sensor_radius=1.0, n_robots=1, n_targets=2, comm_mode="none")
    starts = [(1, 1)]

    m1 = _empty_map(12, 12, starts, [(6, 6), (10, 10)])
    m2 = _empty_map(12, 12, starts, [(6, 6), (2, 10)])  # second target moved elsewhere, still unsensed

    w1 = SwarmWorld(cfg)
    w1.reset(map_spec=m1)
    w2 = SwarmWorld(cfg)
    w2.reset(map_spec=m2)

    g1, v1, mask1 = w1.actor_obs(0)
    g2, v2, mask2 = w2.actor_obs(0)
    assert np.array_equal(g1, g2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(mask1, mask2)


def test_counterfactual_unknown_terrain_invisible_to_actor():
    """Flipping an obstacle in a never-observed region must not change the
    actor's observation."""
    cfg = _cfg(width=12, height=12, obs_radius=3, sensor_radius=1.0, n_robots=1, n_targets=1, comm_mode="none")
    starts = [(1, 1)]
    targets = [(1, 5)]

    rows_a = ["." * 12 for _ in range(12)]
    rows_b = [row for row in rows_a]
    # Flip a cell far from the robot (row 10) to an obstacle — outside obs_radius=3.
    far_row = list(rows_b[10])
    far_row[10] = "#"
    rows_b[10] = "".join(far_row)

    m1 = MapSpec(id="a", width=12, height=12, terrain_rows=rows_a, robot_starts=starts, targets=targets)
    m2 = MapSpec(id="b", width=12, height=12, terrain_rows=rows_b, robot_starts=starts, targets=targets)

    w1 = SwarmWorld(cfg)
    w1.reset(map_spec=m1)
    w2 = SwarmWorld(cfg)
    w2.reset(map_spec=m2)

    g1, v1, mask1 = w1.actor_obs(0)
    g2, v2, mask2 = w2.actor_obs(0)
    assert np.array_equal(g1, g2)
    assert np.array_equal(mask1, mask2)


def test_counterfactual_unsensed_teammate_invisible():
    """Moving an unsensed teammate must not change `o_i` (mode A/B, ch.6)."""
    cfg = _cfg(width=14, height=14, obs_radius=3, sensor_radius=1.0, n_robots=2, n_targets=1, comm_mode="shared_map")
    starts_a = [(1, 1), (12, 12)]
    starts_b = [(1, 1), (12, 5)]  # teammate at a different far-away position
    targets = [(7, 7)]

    m1 = _empty_map(14, 14, starts_a, targets)
    m2 = _empty_map(14, 14, starts_b, targets)

    w1 = SwarmWorld(cfg)
    w1.reset(map_spec=m1)
    w2 = SwarmWorld(cfg)
    w2.reset(map_spec=m2)

    g1, v1, mask1 = w1.actor_obs(0)
    g2, v2, mask2 = w2.actor_obs(0)
    assert np.array_equal(g1, g2)
    assert np.array_equal(v1, v2)


def test_mode_a_isolation_discoveries_outside_sensing_not_shared():
    """In comm_mode='none', robot j's discoveries outside i's sensing must
    not appear in i's belief/observation."""
    cfg = _cfg(width=14, height=14, obs_radius=3, sensor_radius=1.0, n_robots=2, n_targets=1, comm_mode="none")
    starts = [(1, 1), (10, 10)]
    targets = [(10, 11)]  # right next to robot 1, far from robot 0
    m = _empty_map(14, 14, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)  # initial sense: robot 1 discovers the target immediately

    assert world.state.target_status[0] == 1  # DISCOVERED (world truth)
    g0, _, _ = world.actor_obs(0)
    assert np.sum(g0[CH_TARGET]) == 0.0  # robot 0's belief knows nothing about it


def test_mask_local_only_invariant_to_non_sensed_terrain():
    """The action mask must be invariant to terrain changes outside current sensing."""
    cfg = _cfg(width=10, height=10, obs_radius=3, sensor_radius=1.0, n_robots=1, n_targets=1, comm_mode="none")
    starts = [(5, 5)]
    rows_a = ["." * 10 for _ in range(10)]
    rows_b = [row for row in rows_a]
    far_row = list(rows_b[0])
    far_row[0] = "#"
    rows_b[0] = "".join(far_row)  # far from the robot and from the only target reference used here differently

    m1 = MapSpec(id="a", width=10, height=10, terrain_rows=rows_a, robot_starts=starts, targets=[(9, 9)])
    m2 = MapSpec(id="b", width=10, height=10, terrain_rows=rows_b, robot_starts=starts, targets=[(9, 9)])
    w1 = SwarmWorld(cfg)
    w1.reset(map_spec=m1)
    w2 = SwarmWorld(cfg)
    w2.reset(map_spec=m2)
    _, _, mask1 = w1.actor_obs(0)
    _, _, mask2 = w2.actor_obs(0)
    assert np.array_equal(mask1, mask2)


def test_mask_blocks_sensed_obstacle_direction():
    cfg = _cfg(width=6, height=6, obs_radius=3, sensor_radius=1.0, n_robots=1, n_targets=1, comm_mode="none")
    rows = list("." * 6 for _ in range(6))
    row2 = list(rows[2])
    row2[3] = "#"  # obstacle directly right of robot at (2,2)
    rows[2] = "".join(row2)
    m = MapSpec(id="m", width=6, height=6, terrain_rows=rows, robot_starts=[(2, 2)], targets=[(5, 5)])
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    _, _, mask = world.actor_obs(0)
    from marl.envs.core.types import RIGHT

    assert mask[RIGHT] == 0
    assert mask[WAIT] == 1


def test_action_masking_disabled_all_valid_except_bounds_handled_by_dynamics():
    cfg = _cfg(action_masking=False, n_robots=1, n_targets=1)
    world = SwarmWorld(cfg)
    world.reset(seed=1)
    _, _, mask = world.actor_obs(0)
    assert np.all(mask == 1)

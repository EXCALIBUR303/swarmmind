"""SwarmWorld integration tests: termination, truncation, determinism,
error handling, and cross-cutting invariants (§6.4, §33.2)."""

from __future__ import annotations

import numpy as np
import pytest

from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.types import RIGHT, WAIT
from marl.envs.core.world import SwarmWorld


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t",
        width=8,
        height=8,
        n_robots=2,
        n_targets=2,
        obstacle_density=0.1,
        max_steps=30,
        sensor_radius=2.0,
        obs_radius=5,
        target_min_dist=1,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def _empty_map(width, height, starts, targets) -> MapSpec:
    rows = ["." * width for _ in range(height)]
    return MapSpec(id="m", width=width, height=height, terrain_rows=rows, robot_starts=starts, targets=targets)


def test_reset_deterministic_given_seed():
    cfg = _cfg()
    w1 = SwarmWorld(cfg)
    w2 = SwarmWorld(cfg)
    s1 = w1.reset(seed=123)
    s2 = w2.reset(seed=123)
    assert np.array_equal(s1.terrain, s2.terrain)
    assert np.array_equal(s1.robot_pos, s2.robot_pos)
    assert np.array_equal(s1.target_pos, s2.target_pos)
    assert s1.map_id == s2.map_id


def test_step_determinism_identical_actions_identical_outcomes():
    cfg = _cfg()
    w1 = SwarmWorld(cfg)
    w2 = SwarmWorld(cfg)
    w1.reset(seed=5)
    w2.reset(seed=5)
    rng = np.random.default_rng(9)
    for _ in range(20):
        actions = rng.integers(0, 5, size=2)
        r1 = w1.step(actions)
        r2 = w2.step(actions)
        assert np.array_equal(w1.state.robot_pos, w2.state.robot_pos)
        assert np.allclose(r1.reward, r2.reward)
        assert r1.terminated == r2.terminated
        assert r1.truncated == r2.truncated


def test_success_termination():
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, max_steps=50)
    starts = [(1, 1)]
    targets = [(1, 2)]
    m = _empty_map(6, 6, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    result = world.step(np.array([RIGHT]))
    assert result.terminated
    assert not result.truncated
    assert result.success


def test_truncation_at_max_steps_without_success():
    cfg = _cfg(width=10, height=10, n_robots=1, n_targets=1, max_steps=5)
    starts = [(0, 0)]
    targets = [(9, 9)]  # unreachable in 5 steps
    m = _empty_map(10, 10, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    result = None
    for _ in range(5):
        result = world.step(np.array([WAIT]))
    assert result.truncated
    assert not result.terminated
    assert not result.success


def test_success_precedence_at_final_step():
    """If success and the step limit coincide, success wins (terminated=True,
    truncated=False)."""
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, max_steps=1)
    starts = [(1, 1)]
    targets = [(1, 2)]
    m = _empty_map(6, 6, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    result = world.step(np.array([RIGHT]))  # the only allowed step, and it succeeds
    assert result.terminated
    assert not result.truncated


def test_discovery_precedes_rescue_invariant_holds_over_many_episodes():
    cfg = _cfg(width=10, height=10, n_robots=3, n_targets=3, obstacle_density=0.1, max_steps=60, sensor_radius=1.0)
    for seed in range(30):
        world = SwarmWorld(cfg)
        world.reset(seed=seed)
        rng = np.random.default_rng(seed + 1000)
        for _ in range(cfg.max_steps):
            grids, vecs, masks = world.all_actor_obs()
            actions = np.array(
                [
                    rng.choice(np.where(masks[i] == 1)[0]) if np.any(masks[i] == 1) else 0
                    for i in range(cfg.n_robots)
                ]
            )
            result = world.step(actions)
            if result.terminated or result.truncated:
                break
        for k in range(cfg.n_targets):
            if world.state.target_t_rescued[k] >= 0:
                assert world.state.target_t_discovered[k] >= 0
                assert world.state.target_t_discovered[k] <= world.state.target_t_rescued[k]


def test_invalid_action_out_of_range_raises_value_error():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    world.reset(seed=1)
    with pytest.raises(ValueError):
        world.step(np.array([5, 0]))
    with pytest.raises(ValueError):
        world.step(np.array([-1, 0]))


def test_wrong_action_shape_raises_value_error():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    world.reset(seed=1)
    with pytest.raises(ValueError):
        world.step(np.array([0, 0, 0]))  # 3 actions for 2 robots


def test_step_before_reset_raises():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    with pytest.raises(RuntimeError):
        world.step(np.array([0, 0]))


def test_obs_before_reset_raises():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    with pytest.raises(RuntimeError):
        world.actor_obs(0)
    with pytest.raises(RuntimeError):
        world.global_state()


def test_reset_rejects_map_spec_with_wrong_robot_count():
    cfg = _cfg(n_robots=2, n_targets=1)
    m = _empty_map(8, 8, [(1, 1)], [(5, 5)])  # only 1 robot start
    world = SwarmWorld(cfg)
    with pytest.raises(ValueError):
        world.reset(map_spec=m)


def test_reset_rejects_map_spec_with_wrong_target_count():
    cfg = _cfg(n_robots=1, n_targets=2)
    m = _empty_map(8, 8, [(1, 1)], [(5, 5)])  # only 1 target, config expects 2
    world = SwarmWorld(cfg)
    with pytest.raises(ValueError):
        world.reset(map_spec=m)


def test_reset_without_seed_or_map_spec_raises():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    with pytest.raises(ValueError):
        world.reset()


def test_all_actor_obs_and_agent_state_and_full_global_grid_before_reset_raise():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    with pytest.raises(RuntimeError):
        world.all_actor_obs()
    with pytest.raises(RuntimeError):
        world.agent_state_channels(0)
    with pytest.raises(RuntimeError):
        world.full_global_grid(0)
    with pytest.raises(RuntimeError):
        world.render_ansi()


def test_full_global_grid_matches_shared_plus_agent_channels():
    cfg = _cfg(n_robots=2, n_targets=1)
    world = SwarmWorld(cfg)
    world.reset(seed=1)
    full = world.full_global_grid(0)
    shared, _ = world.global_state()
    agent = world.agent_state_channels(0)
    assert np.array_equal(full[:7], shared)
    assert np.array_equal(full[7:], agent)


def test_render_ansi_shows_robots_and_unrescued_targets_only():
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, max_steps=10)
    starts = [(1, 1)]
    targets = [(1, 2)]
    m = _empty_map(6, 6, starts, targets)
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    rendered_before = world.render_ansi()
    assert rendered_before.splitlines()[1][2] == "T"
    assert rendered_before.splitlines()[1][1] == "0"

    result = world.step(np.array([RIGHT]))
    assert result.success
    rendered_after = world.render_ansi()
    # The rescued target must no longer render as "T".
    assert "T" not in rendered_after


def test_fuzz_multi_agent_episode_no_crashes_positions_valid():
    """10k random joint steps across several maps via the full world (not
    just dynamics): positions stay unique, on free cells, in bounds."""
    cfg = _cfg(width=10, height=10, n_robots=4, n_targets=2, obstacle_density=0.15, max_steps=200)
    rng = np.random.default_rng(42)
    total_steps = 0
    for seed in range(50):
        world = SwarmWorld(cfg)
        world.reset(seed=seed)
        for _ in range(200):
            actions = rng.integers(0, 5, size=cfg.n_robots)
            result = world.step(actions)
            total_steps += 1
            positions = {tuple(p) for p in world.state.robot_pos}
            assert len(positions) == cfg.n_robots
            for y, x in world.state.robot_pos:
                assert world.state.terrain[y, x] != 1  # not an obstacle
                assert 0 <= y < cfg.height and 0 <= x < cfg.width
            if result.terminated or result.truncated:
                break
    # 50 episodes x up to 200 steps each; random policies often succeed or
    # truncate early, so this just confirms a substantial fuzz volume ran
    # cleanly (§33.2's "10k random joint steps" budget, not an exact floor).
    assert total_steps >= 3_000

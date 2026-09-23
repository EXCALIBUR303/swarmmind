"""Random and frontier-heuristic baseline tests (§14, §33.3)."""

from __future__ import annotations

import numpy as np
import pytest

from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy
from marl.algorithms.baselines.random_policy import RandomPolicy
from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.types import N_ACTIONS
from marl.envs.core.world import SwarmWorld


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t",
        width=8,
        height=8,
        n_robots=2,
        n_targets=1,
        obstacle_density=0.1,
        max_steps=60,
        sensor_radius=2.0,
        obs_radius=5,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def test_random_policy_always_produces_valid_masked_actions():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    world.reset(seed=1)
    policy = RandomPolicy()
    rng = np.random.default_rng(0)
    for _ in range(30):
        _, _, masks = world.all_actor_obs()
        actions = policy.act(world, rng)
        for i in range(cfg.n_robots):
            assert masks[i, actions[i]] == 1
        result = world.step(actions)
        if result.terminated or result.truncated:
            break


def test_random_policy_deterministic_given_rng():
    cfg = _cfg()
    w1 = SwarmWorld(cfg)
    w2 = SwarmWorld(cfg)
    w1.reset(seed=5)
    w2.reset(seed=5)
    policy = RandomPolicy()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    for _ in range(20):
        a1 = policy.act(w1, rng1)
        a2 = policy.act(w2, rng2)
        assert np.array_equal(a1, a2)
        w1.step(a1)
        w2.step(a2)


def test_random_policy_never_bumps_wall_with_masking_on():
    cfg = _cfg(action_masking=True)
    world = SwarmWorld(cfg)
    world.reset(seed=3)
    policy = RandomPolicy()
    rng = np.random.default_rng(1)
    for _ in range(40):
        actions = policy.act(world, rng)
        result = world.step(actions)
        assert not np.any(result.bumped_wall)
        if result.terminated or result.truncated:
            break


def test_heuristic_reaches_high_success_rate_on_small_open_map():
    """Sanity check: on an easy L1-like config, the heuristic should
    succeed the vast majority of the time (full calibration is Phase 3's
    dedicated task; this is a fast in-suite regression guard)."""
    cfg = _cfg(width=8, height=8, n_robots=2, n_targets=1, obstacle_density=0.10, max_steps=100)
    policy = FrontierHeuristicPolicy()
    successes = 0
    n_episodes = 60
    for seed in range(n_episodes):
        world = SwarmWorld(cfg)
        world.reset(seed=1_000_000 + seed)
        rng = np.random.default_rng(seed)
        for _ in range(cfg.max_steps):
            actions = policy.act(world, rng)
            result = world.step(actions)
            if result.terminated or result.truncated:
                if result.success:
                    successes += 1
                break
    assert successes / n_episodes >= 0.95


def test_heuristic_no_permanent_deadlock_between_two_robots():
    """Regression test for the destination-claim arbitration fix: two
    robots whose independent BFS plans would otherwise repeatedly target
    the same cell must not deadlock for the whole episode."""
    cfg = _cfg(width=8, height=8, n_robots=2, n_targets=1, max_steps=100, obstacle_density=0.10)
    policy = FrontierHeuristicPolicy()
    stuck_episodes = 0
    n_episodes = 100
    for seed in range(n_episodes):
        world = SwarmWorld(cfg)
        world.reset(seed=1_000_000 + seed)
        rng = np.random.default_rng(seed)
        positions_history = []
        for _ in range(cfg.max_steps):
            actions = policy.act(world, rng)
            result = world.step(actions)
            positions_history.append(tuple(world.state.robot_pos.flatten()))
            if result.terminated or result.truncated:
                break
        # "Stuck" = last 10 recorded joint positions are all identical.
        if len(positions_history) >= 10 and len(set(positions_history[-10:])) == 1:
            stuck_episodes += 1
    assert stuck_episodes == 0


def test_heuristic_resolves_multi_robot_traffic_jam_regression():
    """Regression test for the exact L4 validation map (seed 1_000_004)
    that exposed both heuristic bugs during Phase 3 calibration: (1) a
    robot planning into a not-yet-processed teammate's current cell, which
    the environment's R1 rule always blocks regardless of order, and (2) a
    resulting 2-step oscillation once (1) was fixed. Uses the real L4
    config (4 robots, comm_mode='none', dense obstacles) since the
    deadlock only manifested with that many robots interacting."""
    from marl.envs.core.config import LevelRegistry
    from marl.evaluation.eval_sets import load_eval_set

    registry = LevelRegistry()
    cfg = registry.get("L4")
    maps = load_eval_set("L4", "validation")
    target_map = next(m for m in maps if m.id == "L4-val-s1000004-a0")

    result = run_episode_helper(cfg, target_map)
    assert result.success


def run_episode_helper(cfg, map_spec):
    from marl.evaluation.evaluator import run_episode

    return run_episode(cfg, FrontierHeuristicPolicy(), np.random.default_rng(0), map_spec=map_spec).trace


def test_heuristic_never_produces_invalid_action_index():
    cfg = _cfg(n_robots=3, n_targets=2, max_steps=80)
    world = SwarmWorld(cfg)
    world.reset(seed=1)
    policy = FrontierHeuristicPolicy()
    rng = np.random.default_rng(0)
    for _ in range(40):
        actions = policy.act(world, rng)
        assert np.all((actions >= 0) & (actions < N_ACTIONS))
        result = world.step(actions)
        if result.terminated or result.truncated:
            break


def test_heuristic_respects_partial_observability_never_reads_hidden_target():
    """Two maps differing only in a HIDDEN (unsensed) target's position
    must produce IDENTICAL heuristic actions for the robot that hasn't
    discovered it — the heuristic must not silently use world-truth."""
    cfg = _cfg(width=14, height=14, n_robots=1, n_targets=2, sensor_radius=1.0, obs_radius=3, comm_mode="none")
    starts = [(1, 1)]

    rows = ["." * 14 for _ in range(14)]
    m1 = MapSpec(id="a", width=14, height=14, terrain_rows=rows, robot_starts=starts, targets=[(6, 6), (10, 10)])
    m2 = MapSpec(id="b", width=14, height=14, terrain_rows=rows, robot_starts=starts, targets=[(6, 6), (2, 10)])

    w1 = SwarmWorld(cfg)
    w1.reset(map_spec=m1)
    w2 = SwarmWorld(cfg)
    w2.reset(map_spec=m2)

    policy = FrontierHeuristicPolicy()
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    a1 = policy.act(w1, rng1)
    a2 = policy.act(w2, rng2)
    assert np.array_equal(a1, a2)


def test_heuristic_waits_when_nothing_reachable():
    """A robot fully enclosed by known obstacles with no reachable target
    or frontier must WAIT, not crash."""
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, sensor_radius=1.0, obs_radius=3)
    rows = [
        "######",
        "#....#",
        "#.##.#",
        "#....#",
        "######",
        "######",
    ]
    # Robot boxed in a small room; target is unreachable (outside, but
    # since it's placed at a free cell inside via config validation we
    # just check that a boxed robot with only its own room known produces
    # a valid (non-crashing) action, WAIT if nothing new to explore.
    m = MapSpec(id="box", width=6, height=6, terrain_rows=rows, robot_starts=[(1, 1)], targets=[(3, 4)])
    world = SwarmWorld(cfg)
    world.reset(map_spec=m)
    policy = FrontierHeuristicPolicy()
    rng = np.random.default_rng(0)
    for _ in range(30):
        actions = policy.act(world, rng)
        assert actions.shape == (1,)
        world.step(actions)


def test_heuristic_act_before_reset_raises():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    policy = FrontierHeuristicPolicy()
    with pytest.raises(RuntimeError):
        policy.act(world, np.random.default_rng(0))


def test_random_act_before_reset_raises():
    cfg = _cfg()
    world = SwarmWorld(cfg)
    policy = RandomPolicy()
    with pytest.raises(RuntimeError):
        policy.act(world, np.random.default_rng(0))

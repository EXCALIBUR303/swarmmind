"""Coordination-episode running: the isolation-null runner and the
team/isolation aggregation (§24.3, §33.3 determinism)."""

from __future__ import annotations

import pytest

from marl.algorithms.baselines.random_policy import RandomPolicy
from marl.envs.core.config import EnvConfig
from marl.envs.core.mapgen import generate_map
from marl.evaluation.coordination import (
    aggregate_coordination,
    isolation_map_and_config,
    run_coordination_episode,
    run_isolation_null,
)


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t", width=8, height=8, n_robots=3, n_targets=2, obstacle_density=0.1,
        max_steps=40, sensor_radius=2.0, obs_radius=5,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def test_isolation_map_and_config_keeps_terrain_and_targets_but_one_robot():
    cfg = _cfg()
    map_spec = generate_map(cfg, seed=1)
    iso_cfg, iso_map = isolation_map_and_config(cfg, map_spec, robot_idx=1)
    assert iso_cfg.n_robots == 1
    assert iso_cfg.n_targets == cfg.n_targets  # untouched
    assert iso_map.terrain_rows == map_spec.terrain_rows
    assert iso_map.targets == map_spec.targets
    assert iso_map.robot_starts == [map_spec.robot_starts[1]]
    # The world accepts it (n_robots/n_targets now match the map).
    from marl.envs.core.world import SwarmWorld

    world = SwarmWorld(iso_cfg)
    world.reset(map_spec=iso_map)  # must not raise


def test_run_isolation_null_is_deterministic():
    cfg = _cfg()
    map_spec = generate_map(cfg, seed=2)
    a = run_isolation_null(cfg, RandomPolicy, map_spec, eval_seed=0, n_robots=cfg.n_robots)
    b = run_isolation_null(cfg, RandomPolicy, map_spec, eval_seed=0, n_robots=cfg.n_robots)
    assert a == b
    assert 0.0 <= a <= (cfg.n_robots - 1) / cfg.n_robots + 1e-9


def test_run_isolation_null_differs_from_a_different_eval_seed_usually():
    cfg = _cfg()
    map_spec = generate_map(cfg, seed=3)
    values = {run_isolation_null(cfg, RandomPolicy, map_spec, eval_seed=s, n_robots=cfg.n_robots) for s in range(5)}
    assert len(values) > 1  # stochastic Random policy shouldn't give one constant DER_iso


def test_run_coordination_episode_returns_metrics_and_der_iso():
    cfg = _cfg()
    map_spec = generate_map(cfg, seed=4)
    metrics, der_iso = run_coordination_episode(cfg, RandomPolicy, map_spec, eval_seed=0)
    assert der_iso is not None and 0.0 <= der_iso <= 1.0
    assert 0.0 <= metrics.der <= 1.0


def test_run_coordination_episode_skips_isolation_when_requested_or_n_1():
    cfg = _cfg()
    map_spec = generate_map(cfg, seed=5)
    _metrics, der_iso = run_coordination_episode(cfg, RandomPolicy, map_spec, eval_seed=0, with_isolation=False)
    assert der_iso is None

    cfg1 = _cfg(n_robots=1, n_targets=1)
    map1 = generate_map(cfg1, seed=6)
    _metrics, der_iso1 = run_coordination_episode(cfg1, RandomPolicy, map1, eval_seed=0)
    assert der_iso1 is None  # nothing to isolate from with a single robot


def test_aggregate_coordination_reports_excluded_counts_for_undefined_fields():
    from marl.analysis.coordination import CoordinationMetrics

    episodes = [
        CoordinationMetrics(der=0.1, der_at_50=0.2, visitation_overlap=0.0, redundant_pursuit=1.0,
                            allocation_optimality=1.0, mean_inter_robot_distance=3.0,
                            min_inter_robot_distance=2.0, dispersion_speed=5),
        CoordinationMetrics(der=0.3, der_at_50=None, visitation_overlap=0.1, redundant_pursuit=None,
                            allocation_optimality=None, mean_inter_robot_distance=4.0,
                            min_inter_robot_distance=1.0, dispersion_speed=None),
    ]
    agg = aggregate_coordination(episodes, der_iso_values=[0.4, 0.6])
    assert agg["n_episodes"] == 2
    assert agg["der"] == pytest.approx(0.2)
    assert agg["der_at_50"] == {"mean": pytest.approx(0.2), "n_defined": 1, "n_excluded": 1}
    assert agg["redundant_pursuit"]["n_defined"] == 1 and agg["redundant_pursuit"]["n_excluded"] == 1
    assert agg["der_iso"] == pytest.approx(0.5)


def test_aggregate_coordination_all_undefined_gives_none_mean():
    from marl.analysis.coordination import CoordinationMetrics

    episodes = [
        CoordinationMetrics(der=0.0, der_at_50=None, visitation_overlap=0.0, redundant_pursuit=None,
                            allocation_optimality=None, mean_inter_robot_distance=0.0,
                            min_inter_robot_distance=0.0, dispersion_speed=None),
    ]
    agg = aggregate_coordination(episodes, der_iso_values=[])
    assert agg["der_at_50"]["mean"] is None
    assert agg["der_iso"] is None


def test_isolation_null_never_touches_more_than_one_robot_per_run():
    """A structural guard: for N robots, exactly N single-robot episodes
    are run (never fewer, never a joint N-robot run)."""
    calls: list[int] = []
    cfg = _cfg()
    map_spec = generate_map(cfg, seed=7)

    class CountingRandomPolicy(RandomPolicy):
        def act(self, world, rng):
            calls.append(world.state.n_robots)
            return super().act(world, rng)

    run_isolation_null(cfg, CountingRandomPolicy, map_spec, eval_seed=0, n_robots=cfg.n_robots)
    assert calls and all(n == 1 for n in calls)


def test_isolation_null_uses_the_frozen_l2_structured_maps_without_error():
    """Sanity check against real, committed maps (not just procedural ones)."""
    from marl.envs.core.config import LevelRegistry
    from marl.evaluation.eval_sets import load_eval_set

    env_cfg = LevelRegistry().get("L2")
    maps = load_eval_set("L2", "structured")
    der_iso = run_isolation_null(env_cfg, RandomPolicy, maps[0], eval_seed=0, n_robots=env_cfg.n_robots)
    assert 0.0 <= der_iso <= 1.0

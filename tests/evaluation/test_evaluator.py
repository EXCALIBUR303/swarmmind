"""Episode-runner (evaluator) tests: reproducibility and trace correctness (§33.4)."""

from __future__ import annotations

import numpy as np

from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy
from marl.algorithms.baselines.random_policy import RandomPolicy
from marl.envs.core.config import EnvConfig
from marl.evaluation.evaluator import run_episode


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


def test_run_episode_reproducible_given_seed_and_rng():
    cfg = _cfg()
    r1 = run_episode(cfg, RandomPolicy(), np.random.default_rng(3), seed=7)
    r2 = run_episode(cfg, RandomPolicy(), np.random.default_rng(3), seed=7)
    assert r1.trace.actions[0].tolist() == r2.trace.actions[0].tolist()
    assert len(r1.trace.actions) == len(r2.trace.actions)
    assert r1.trace.success == r2.trace.success
    assert r1.trace.final_t == r2.trace.final_t


def test_run_episode_trace_length_matches_steps_taken():
    cfg = _cfg(max_steps=10)
    result = run_episode(cfg, RandomPolicy(), np.random.default_rng(0), seed=1)
    n_steps = len(result.trace.actions)
    assert len(result.trace.positions) == n_steps + 1
    assert len(result.trace.rewards) == n_steps
    assert len(result.trace.coverage_over_time) == n_steps + 1
    assert n_steps <= cfg.max_steps


def test_run_episode_heuristic_success_on_easy_map():
    cfg = _cfg(width=8, height=8, n_robots=2, n_targets=1, obstacle_density=0.1, max_steps=100)
    result = run_episode(cfg, FrontierHeuristicPolicy(), np.random.default_rng(0), seed=1_000_000)
    assert result.trace.success
    assert result.n_reachable_free > 0


def test_run_episode_truncates_without_success_when_unreachable():
    from marl.envs.core.maps_io import MapSpec

    cfg = _cfg(width=10, height=10, n_robots=1, n_targets=1, max_steps=3)
    rows = ["." * 10 for _ in range(10)]
    m = MapSpec(id="m", width=10, height=10, terrain_rows=rows, robot_starts=[(0, 0)], targets=[(9, 9)])
    result = run_episode(cfg, RandomPolicy(), np.random.default_rng(0), map_spec=m)
    assert not result.trace.success
    assert result.trace.final_t == 3
    assert len(result.trace.actions) == 3


def test_run_episode_map_id_matches_world():
    cfg = _cfg()
    result = run_episode(cfg, RandomPolicy(), np.random.default_rng(0), seed=42)
    assert result.trace.map_id.startswith("t-s42")

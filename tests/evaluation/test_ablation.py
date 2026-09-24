"""Input-ablation evaluation (§22.4 item 4, §24.4 criterion 4)."""

from __future__ import annotations

import pytest
import torch

from marl.algorithms.common.config import AlgoConfig
from marl.algorithms.ippo.agent import IPPOAgent
from marl.envs.core.config import LevelRegistry
from marl.envs.core.mapgen import generate_map
from marl.evaluation.ablation import run_ablation_for_checkpoint, summarize_ablation


def _actor(seed: int = 0):
    torch.manual_seed(seed)
    cfg = AlgoConfig.model_validate({"name": "ippo", "critic": {"input": "local"}})
    return IPPOAgent(cfg, grid_size=11).actor


def test_run_ablation_for_checkpoint_has_baseline_and_both_channels():
    env_cfg = LevelRegistry().get("L1")
    maps = [generate_map(env_cfg, seed=1)]
    result = run_ablation_for_checkpoint(env_cfg, _actor(), maps, eval_seeds=[0])
    assert set(result) == {"baseline", "channel5_visited_known", "channel6_teammate"}
    for arm in result.values():
        assert 0.0 <= arm["success_rate"] <= 1.0
        assert arm["collisions"] >= 0.0


def test_summarize_ablation_computes_expected_medians():
    per_seed = [
        {"baseline": {"success_rate": 1.0, "collisions": 0.1}, "channel5_visited_known": {"success_rate": 0.9, "collisions": 0.1},
         "channel6_teammate": {"success_rate": 0.9, "collisions": 10.0}},
        {"baseline": {"success_rate": 1.0, "collisions": 0.2}, "channel5_visited_known": {"success_rate": 0.8, "collisions": 0.2},
         "channel6_teammate": {"success_rate": 0.9, "collisions": 20.0}},
        {"baseline": {"success_rate": 1.0, "collisions": 0.1}, "channel5_visited_known": {"success_rate": 1.0, "collisions": 0.1},
         "channel6_teammate": {"success_rate": 0.9, "collisions": 30.0}},
    ]
    summary = summarize_ablation(per_seed)
    assert summary["n_seeds"] == 3
    assert summary["channel5_success_drop_per_seed"] == pytest.approx([0.1, 0.2, 0.0])
    assert summary["channel5_success_drop_median"] == pytest.approx(0.1)
    assert summary["channel6_collision_multiplier_per_seed"] == [100.0, 100.0, 300.0]
    assert summary["channel6_collision_multiplier_median"] == 100.0


def test_summarize_ablation_handles_zero_baseline_collisions_without_dividing_by_zero():
    per_seed = [
        {"baseline": {"success_rate": 1.0, "collisions": 0.0}, "channel5_visited_known": {"success_rate": 1.0, "collisions": 0.0},
         "channel6_teammate": {"success_rate": 0.9, "collisions": 5.0}},
    ]
    summary = summarize_ablation(per_seed)
    assert summary["channel6_collision_multiplier_median"] == 250.0  # 5.0 / floor(0.02)


def test_summarize_ablation_empty_input():
    summary = summarize_ablation([])
    assert summary["channel6_collision_multiplier_median"] is None
    assert summary["channel5_success_drop_median"] is None

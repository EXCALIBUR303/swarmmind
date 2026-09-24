"""MAPPO learning smoke test (§33.3 (b)/(c)), scaled down like the IPPO one.

The full-scale Phase 5 gate (3 seeds x ~300k env steps on L1, same config
as the Phase 4 IPPO gate) is run via `scripts/run_l1_gate.py` and reported
in `docs/phase_reports/PHASE_5.md`; this checks the mechanism quickly.
"""

from __future__ import annotations

import pytest

from marl.algorithms.common.config import AlgoConfig
from marl.algorithms.mappo.agent import MAPPOAgent
from marl.envs.core.config import LevelRegistry
from marl.networks.actor import ActorPolicy
from marl.training.checkpoint import load_checkpoint
from marl.training.trainer import train
from marl.training.validate import evaluate_on_validation

pytestmark = pytest.mark.slow


def test_mappo_l1_reduced_budget_improves_over_untrained(tmp_path):
    """Budget sized from the measured Phase 5 learning curves: at 20 updates
    of 16 envs (~20k env steps) all three full-scale MAPPO seeds were at
    0.84-0.96 validation success vs ~0.6 untrained. (A first version used
    25 updates of 8 envs judged on 12 maps — below where either IPPO or
    MAPPO reliably separates from its untrained baseline at 1/12
    granularity — and tied 0.58 vs 0.58.)"""
    env_cfg = LevelRegistry().get("L1")
    algo_cfg = AlgoConfig.model_validate({"name": "mappo", "critic": {"input": "agent_specific_global"}})
    algo_cfg.ppo.n_envs = 16
    algo_cfg.ppo.rollout_len = 64
    algo_cfg.eval.interval_updates = 1000
    algo_cfg.eval.n_val_maps = 25
    algo_cfg.checkpoint.fractions = [0.0, 1.0]

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    result = train(
        env_cfg, algo_cfg, MAPPOAgent, run_seed=0, total_env_steps=16 * 64 * 20,
        run_dir=run_dir, study="smoke", algo_name="mappo", level="L1",
    )
    assert result.status == "completed"
    assert result.final_val_metrics is not None

    actor0 = ActorPolicy(grid_size=11, hidden=256)
    actor0.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "step_0.pt")["actor_state_dict"])
    untrained = evaluate_on_validation(env_cfg, actor0, "L1", n_maps=25, stochastic=True)
    assert result.final_val_metrics["success_rate"] > untrained["success_rate"]

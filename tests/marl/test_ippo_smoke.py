"""IPPO learning smoke tests (§33.3 (a)/(c), Phase 4 validation gate).

These are SCALED-DOWN, fast versions of the spec's smoke tests, sized to
finish in well under a minute each so they're practical to run repeatedly
(§33.1: "learning smoke tests, minutes" — these are seconds). They check
the qualitative claim (the policy demonstrably improves over its
untrained baseline) rather than the full quantitative thresholds, which
were verified once, at full scale (200k-300k env steps, matching the
spec's exact configs), via ad hoc scripts and are reported in
`docs/phase_reports/PHASE_4.md`:
  - smoke (a): 6x6, 1 robot, 1 target, density 0 -> 100% validation
    success within 200k env steps (gate: >=90%).
  - L1 (2 robots, 8x8) -> 3/3 seeds reached success >=0.96 and censored
    mean steps <=10.7 within 300k env steps (gate, from Phase 3:
    success>=0.85, steps<=50.0).

Re-running those exact full-scale configs here on every test invocation
would take 10-20+ minutes total, so this file re-validates the same
*mechanism* (does training measurably improve the policy at all) at a
budget that fits a normal test run.
"""

from __future__ import annotations

import pytest
import torch

from marl.algorithms.common.config import AlgoConfig
from marl.algorithms.ippo.agent import IPPOAgent
from marl.envs.core.config import EnvConfig
from marl.training.trainer import train

pytestmark = pytest.mark.slow


def _tiny_smoke_cfg() -> tuple[EnvConfig, AlgoConfig]:
    env_cfg = EnvConfig(
        name="smoke_a",
        width=6,
        height=6,
        n_robots=1,
        n_targets=1,
        obstacle_density=0.0,
        max_steps=40,
        sensor_radius=2.0,
        obs_radius=5,
    )
    algo_cfg = AlgoConfig.model_validate({"name": "ippo", "critic": {"input": "local"}})
    algo_cfg.ppo.n_envs = 8
    algo_cfg.ppo.rollout_len = 64
    algo_cfg.eval.interval_updates = 1000  # only evaluate at the final update
    algo_cfg.eval.n_val_maps = 12
    algo_cfg.checkpoint.fractions = [0.0, 1.0]
    return env_cfg, algo_cfg


def test_ippo_smoke_a_tiny_env_learns_measurably(tmp_path):
    """A scaled-down version of smoke test (a): a single robot on an open
    6x6 map should improve substantially over its untrained baseline
    within a small budget. Full-scale (200k steps) result: 100% success,
    see docs/phase_reports/PHASE_4.md."""
    env_cfg, algo_cfg = _tiny_smoke_cfg()
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    total_env_steps = 8 * 64 * 20  # 20 updates, ~10k env steps

    result = train(
        env_cfg, algo_cfg, IPPOAgent, run_seed=0, total_env_steps=total_env_steps,
        run_dir=run_dir, study="smoke", algo_name="ippo", level="smoke_a",
    )

    assert result.status == "completed"
    assert result.final_val_metrics is not None
    # A generous but meaningful bar for this reduced budget: comfortably
    # better than the untrained baseline (empirically ~0.55-0.65 on this
    # config, matching Random) and clearly trending toward solved.
    assert result.final_val_metrics["success_rate"] >= 0.7


def test_ippo_zero_percent_checkpoint_matches_random_level():
    """The untrained (0% fraction) checkpoint's validation performance
    must be close to what an untrained/random policy achieves — i.e.
    training must not have "cheated" by starting from a lucky init, and
    the checkpoint-at-0% mechanism must genuinely capture the untrained
    policy (§43 Phase 4 gate: '0% ckpt ≈ random')."""
    import numpy as np

    from marl.algorithms.baselines.random_policy import RandomPolicy
    from marl.evaluation.evaluator import run_episode
    from marl.training.validate import evaluate_on_validation

    env_cfg, algo_cfg = _tiny_smoke_cfg()
    torch.manual_seed(0)
    agent = IPPOAgent(algo_cfg, grid_size=11)

    untrained_metrics = evaluate_on_validation(env_cfg, agent.actor, "smoke_a", n_maps=12, stochastic=True)

    random_successes = 0
    for i in range(12):
        rng = np.random.default_rng(1_000_000 + i)
        result = run_episode(env_cfg, RandomPolicy(), rng, seed=1_000_000 + i)
        random_successes += int(result.trace.success)
    random_success_rate = random_successes / 12

    # Both should be in a similar, low-success ballpark (untrained networks
    # act close to randomly under a near-uniform initial policy, §15.2).
    assert abs(untrained_metrics["success_rate"] - random_success_rate) < 0.35


def test_ippo_l1_gate_reduced_budget_shows_clear_improvement(tmp_path):
    """A reduced-budget run on the REAL L1 config (2 robots, 8x8 — the
    actual Phase 4 gate level) that checks directional improvement rather
    than the full threshold, since the full threshold needs ~100k+ steps
    (verified separately at full scale, see docs/phase_reports/PHASE_4.md:
    3/3 seeds reached success>=0.96 at 300k steps).

    Budget sized from the clean Phase 5 IPPO curves: at ~30k env steps all
    three seeds were at 0.84-0.96 vs ~0.6 untrained (at 20k one seed was
    still at 0.66). The original 25 updates x 8 envs judged on 12 maps sat
    below that and passed in Phase 4 only by luck of the (buggy, v1) map
    stream — it failed once mapgen v2 changed which maps were drawn."""
    from marl.envs.core.config import LevelRegistry

    env_cfg = LevelRegistry().get("L1")
    algo_cfg = AlgoConfig.model_validate({"name": "ippo", "critic": {"input": "local"}})
    algo_cfg.ppo.n_envs = 16
    algo_cfg.ppo.rollout_len = 64
    algo_cfg.eval.interval_updates = 1000
    algo_cfg.eval.n_val_maps = 25
    algo_cfg.checkpoint.fractions = [0.0, 1.0]

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    total_env_steps = 16 * 64 * 30  # ~30k env steps

    result = train(
        env_cfg, algo_cfg, IPPOAgent, run_seed=0, total_env_steps=total_env_steps,
        run_dir=run_dir, study="smoke", algo_name="ippo", level="L1",
    )

    assert result.status == "completed"
    from marl.networks.actor import ActorPolicy
    from marl.training.checkpoint import load_checkpoint
    from marl.training.validate import evaluate_on_validation

    ckpt0 = load_checkpoint(run_dir / "checkpoints" / "step_0.pt")
    actor0 = ActorPolicy(grid_size=11, hidden=256)
    actor0.load_state_dict(ckpt0["actor_state_dict"])
    untrained = evaluate_on_validation(env_cfg, actor0, "L1", n_maps=25, stochastic=True)

    assert result.final_val_metrics["success_rate"] > untrained["success_rate"]

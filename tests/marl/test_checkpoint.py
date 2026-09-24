"""Checkpoint save/load/resume and actor-only inference tests (§15.10, §33.3)."""

from __future__ import annotations

import numpy as np
import torch

from marl.algorithms.common.config import AlgoConfig
from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
from marl.algorithms.ippo.agent import IPPOAgent
from marl.envs.core.config import LevelRegistry
from marl.training.checkpoint import (
    export_actor,
    load_actor_export,
    load_checkpoint,
    restore_agent,
    save_checkpoint,
)
from marl.training.trainer import train
from marl.utils.config_loader import load_yaml
from marl.utils.paths import CONFIGS_DIR


def _make_agent(seed: int = 0) -> IPPOAgent:
    torch.manual_seed(seed)
    cfg = AlgoConfig.model_validate(load_yaml(CONFIGS_DIR / "algo" / "ippo.yaml"))
    return IPPOAgent(cfg, grid_size=11)


def test_checkpoint_save_load_round_trip(tmp_path):
    agent = _make_agent()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, agent, global_env_steps=1000, update_idx=5, resolved_config={"a": 1}, config_hash="abcd1234")

    ckpt = load_checkpoint(path)
    assert ckpt["global_env_steps"] == 1000
    assert ckpt["update_idx"] == 5
    assert ckpt["config_hash"] == "abcd1234"
    assert "actor_state_dict" in ckpt
    assert "critic_state_dict" in ckpt
    assert "rng_state" in ckpt


def test_restore_agent_recovers_exact_weights(tmp_path):
    agent1 = _make_agent(seed=1)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, agent1, 0, 0, {}, "h")

    agent2 = _make_agent(seed=2)  # different init
    ckpt = load_checkpoint(path)
    restore_agent(agent2, ckpt)

    for p1, p2 in zip(agent1.actor.parameters(), agent2.actor.parameters(), strict=True):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(agent1.critic.parameters(), agent2.critic.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_actor_only_export_contains_no_critic_state(tmp_path):
    agent = _make_agent()
    path = tmp_path / "actor.pt"
    export_actor(path, agent, meta={"grid_size": 11, "hidden": 256})

    payload = load_actor_export(path)
    assert "critic_state_dict" not in payload
    assert "opt_actor_state_dict" not in payload
    assert "opt_critic_state_dict" not in payload
    assert set(payload.keys()) == {"schema_version", "actor_state_dict", "meta"}


def test_actor_only_export_runs_inference_without_critic_import(tmp_path):
    agent = _make_agent()
    path = tmp_path / "actor.pt"
    export_actor(path, agent, meta={"grid_size": 11, "hidden": 256})

    actor, meta = load_actor(str(path))
    grid = torch.rand(1, 8, 11, 11)
    vec = torch.rand(1, 10)
    mask = torch.ones(1, 5, dtype=torch.bool)
    actions, _ = actor.act(grid, vec, mask)
    assert actions.shape == (1,)
    assert meta["grid_size"] == 11


def test_learned_actor_policy_runs_full_episode():
    agent = _make_agent()
    policy = LearnedActorPolicy(agent.actor, stochastic=True)
    env_cfg = LevelRegistry().get("L1")

    from marl.envs.core.world import SwarmWorld

    world = SwarmWorld(env_cfg)
    world.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        actions = policy.act(world, rng)
        assert actions.shape == (env_cfg.n_robots,)
        result = world.step(actions)
        if result.terminated or result.truncated:
            break


def test_learned_actor_policy_reproducible_given_rng_stream():
    agent = _make_agent()
    policy = LearnedActorPolicy(agent.actor, stochastic=True)
    env_cfg = LevelRegistry().get("L1")

    from marl.envs.core.world import SwarmWorld

    w1, w2 = SwarmWorld(env_cfg), SwarmWorld(env_cfg)
    w1.reset(seed=3)
    w2.reset(seed=3)
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(10):
        a1 = policy.act(w1, rng1)
        a2 = policy.act(w2, rng2)
        assert np.array_equal(a1, a2)
        w1.step(a1)
        w2.step(a2)


def test_resume_equivalence_bitwise_on_parameters(tmp_path):
    """§33.3: 4 updates straight must equal 2 updates + save + resume + 2
    updates, bitwise on parameters (CPU)."""
    env_cfg = LevelRegistry().get("L1")
    algo_cfg = AlgoConfig.model_validate(load_yaml(CONFIGS_DIR / "algo" / "ippo.yaml"))
    algo_cfg.ppo.n_envs = 4
    algo_cfg.ppo.rollout_len = 16
    algo_cfg.eval.interval_updates = 1000
    # Small but non-zero: the split run validates at points the straight run
    # doesn't (after its first half's last update), so bitwise-equal
    # parameters also show validation never perturbs the training RNG streams.
    algo_cfg.eval.n_val_maps = 3
    algo_cfg.checkpoint.fractions = [0.0, 1.0]

    total_env_steps = 4 * 16 * 4  # 4 updates

    run_a = tmp_path / "a"
    (run_a / "checkpoints").mkdir(parents=True)
    result_a = train(
        env_cfg, algo_cfg, IPPOAgent, run_seed=11, total_env_steps=total_env_steps,
        run_dir=run_a, study="t", algo_name="ippo", level="L1",
    )

    run_b = tmp_path / "b"
    (run_b / "checkpoints").mkdir(parents=True)
    half_steps = 4 * 16 * 2  # 2 updates
    train(
        env_cfg, algo_cfg, IPPOAgent, run_seed=11, total_env_steps=half_steps,
        run_dir=run_b, study="t", algo_name="ippo", level="L1",
    )
    result_b = train(
        env_cfg, algo_cfg, IPPOAgent, run_seed=11, total_env_steps=total_env_steps,
        run_dir=run_b, study="t", algo_name="ippo", level="L1",
        resume_from=run_b / "checkpoints" / "latest.pt",
    )

    ckpt_a = load_checkpoint(run_a / "checkpoints" / "latest.pt")
    ckpt_b = load_checkpoint(run_b / "checkpoints" / "latest.pt")

    assert result_a.global_env_steps == result_b.global_env_steps
    for key in ckpt_a["actor_state_dict"]:
        assert torch.equal(ckpt_a["actor_state_dict"][key], ckpt_b["actor_state_dict"][key]), key
    for key in ckpt_a["critic_state_dict"]:
        assert torch.equal(ckpt_a["critic_state_dict"][key], ckpt_b["critic_state_dict"][key]), key


def test_final_checkpoint_saved_when_budget_not_multiple_of_update(tmp_path):
    """Regression (Phase 6): with a budget that isn't a multiple of E*T
    (e.g. 3M / 2048), fraction 1.0 used to be measured against the nominal
    budget, never reached, and the final checkpoint was never saved."""
    import json

    env_cfg = LevelRegistry().get("L1")
    algo_cfg = AlgoConfig.model_validate(load_yaml(CONFIGS_DIR / "algo" / "ippo.yaml"))
    algo_cfg.ppo.n_envs = 2
    algo_cfg.ppo.rollout_len = 8
    algo_cfg.eval.interval_updates = 1000
    algo_cfg.eval.n_val_maps = 1
    algo_cfg.checkpoint.fractions = [0.0, 0.5, 1.0]
    run = tmp_path / "r"
    (run / "checkpoints").mkdir(parents=True)
    train(env_cfg, algo_cfg, IPPOAgent, run_seed=0, total_env_steps=16 * 4 + 5,  # 4 updates + remainder
          run_dir=run, study="t", algo_name="ippo", level="L1")
    manifest = json.loads((run / "manifest.json").read_text())
    by_frac = {f: c["env_steps"] for c in manifest["checkpoints"] for f in c["fractions"]}
    assert by_frac == {0.0: 0, 0.5: 32, 1.0: 64}
    assert manifest["trained_env_steps"] == 64 and manifest["budget_env_steps"] == 69

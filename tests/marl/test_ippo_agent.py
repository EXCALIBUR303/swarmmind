"""IPPOAgent construction and forward-pass tests (§15.1, §33.3)."""

from __future__ import annotations

import pytest
import torch

from marl.algorithms.common.agent import CriticInputs
from marl.algorithms.common.config import AlgoConfig
from marl.algorithms.ippo.agent import IPPOAgent


def _cfg(**overrides) -> AlgoConfig:
    base = AlgoConfig.model_validate({"name": "ippo", "critic": {"input": "local"}})
    return base.model_copy(update=overrides) if overrides else base


def test_ippo_agent_rejects_mappo_config():
    cfg = AlgoConfig.model_validate({"name": "mappo", "critic": {"input": "agent_specific_global"}})
    with pytest.raises(ValueError):
        IPPOAgent(cfg)


def test_ippo_critic_input_uses_only_local_observation():
    agent = IPPOAgent(_cfg())
    grid = torch.rand(3, 8, 11, 11)
    vec = torch.rand(3, 10)
    inputs = CriticInputs(local_grid=grid, local_vec=vec)
    value = agent.critic_value_denormalized(inputs)
    assert value.shape == (3,)


def test_ippo_collect_returns_actions_logp_value():
    agent = IPPOAgent(_cfg())
    grid = torch.rand(2, 8, 11, 11)
    vec = torch.rand(2, 10)
    mask = torch.ones(2, 5, dtype=torch.bool)
    inputs = CriticInputs(local_grid=grid, local_vec=vec)
    gen = torch.Generator().manual_seed(0)
    actions, logp, value = agent.collect(grid, vec, mask, inputs, gen, stochastic=True)
    assert actions.shape == (2,)
    assert logp.shape == (2,)
    assert value.shape == (2,)


def test_ippo_evaluate_has_gradients():
    agent = IPPOAgent(_cfg())
    grid = torch.rand(2, 8, 11, 11)
    vec = torch.rand(2, 10)
    mask = torch.ones(2, 5, dtype=torch.bool)
    inputs = CriticInputs(local_grid=grid, local_vec=vec)
    dist, value_n = agent.evaluate(grid, vec, mask, inputs)
    loss = -dist.log_prob(torch.zeros(2, dtype=torch.long)).mean() + value_n.mean()
    loss.backward()
    assert agent.actor.policy_head.weight.grad is not None
    assert agent.critic.value_head.weight.grad is not None


def test_ippo_value_norm_enabled_by_default():
    agent = IPPOAgent(_cfg())
    assert agent.value_norm is not None


def test_ippo_value_norm_can_be_disabled():
    cfg = _cfg()
    cfg.ppo.value_norm = False
    agent = IPPOAgent(cfg)
    assert agent.value_norm is None
    grid = torch.rand(1, 8, 11, 11)
    vec = torch.rand(1, 10)
    inputs = CriticInputs(local_grid=grid, local_vec=vec)
    # denormalized == raw output when value_norm is off
    raw = agent._critic_forward(inputs)
    denorm = agent.critic_value_denormalized(inputs)
    assert torch.equal(raw, denorm)

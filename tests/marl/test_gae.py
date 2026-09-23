"""GAE tests: hand-computed cases, truncation folding, no cross-agent/env
leakage, advantage normalization (§15.5-15.6, §33.3)."""

from __future__ import annotations

import torch

from marl.algorithms.common.gae import compute_gae, normalize_advantages


def test_gae_hand_computed_termination_case():
    """T=2,E=1,N=1: episode terminates at t=1. Hand-computed expected values
    (gamma=0.99, lambda=0.95): adv=[1.46525, 0.5], ret=[1.96525, 1.0]."""
    rewards = torch.tensor([[[1.0]], [[1.0]]])
    values = torch.tensor([[[0.5]], [[0.5]]])
    dones = torch.tensor([[False], [True]])
    last_value = torch.tensor([[0.0]])  # irrelevant: not_done=0 zeroes it out

    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95)
    assert torch.allclose(adv.flatten(), torch.tensor([1.46525, 0.5]), atol=1e-4)
    assert torch.allclose(ret.flatten(), torch.tensor([1.96525, 1.0]), atol=1e-4)


def test_gae_hand_computed_rollout_end_bootstrap_case():
    """T=2,E=1,N=1: rollout ends WITHOUT termination -> bootstrap from
    last_value. gamma=0.9, lambda=0.9.
    delta_1 = r1 + gamma*last_value - v1 = 1 + 0.9*2.0 - 0.5 = 2.3
    adv_1 = 2.3
    delta_0 = r0 + gamma*v1 - v0 = 1 + 0.9*0.5 - 0.5 = 0.95
    adv_0 = 0.95 + gamma*lambda*adv_1 = 0.95 + 0.81*2.3 = 2.813
    """
    rewards = torch.tensor([[[1.0]], [[1.0]]])
    values = torch.tensor([[[0.5]], [[0.5]]])
    dones = torch.tensor([[False], [False]])  # not done at rollout end -> bootstrap
    last_value = torch.tensor([[2.0]])

    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.9, gae_lambda=0.9)
    assert torch.allclose(adv.flatten(), torch.tensor([2.813, 2.3]), atol=1e-3)
    assert torch.allclose(ret.flatten(), torch.tensor([3.313, 2.8]), atol=1e-3)


def test_gae_truncation_folding_case():
    """Truncation folding means: by the time compute_gae is called, the
    caller has ALREADY added gamma*v_final into the reward at the
    truncated step, and passes done=True for it. This must produce the
    SAME result as the termination case (since the mechanism is
    identical from GAE's point of view)."""
    gamma, lam = 0.99, 0.95
    v_final = 3.0
    raw_reward_at_trunc = 1.0
    folded_reward = raw_reward_at_trunc + gamma * v_final

    rewards = torch.tensor([[[1.0]], [[folded_reward]]])
    values = torch.tensor([[[0.5]], [[0.5]]])
    dones = torch.tensor([[False], [True]])  # truncation folded -> treated as done
    last_value = torch.tensor([[0.0]])

    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=gamma, gae_lambda=lam)
    # delta_1 = folded_reward - v1 (no further bootstrap since not_done=0)
    expected_delta_1 = folded_reward - 0.5
    assert torch.allclose(adv[1].flatten(), torch.tensor([expected_delta_1]), atol=1e-5)


def test_gae_no_leakage_across_agents():
    """A unique reward on one agent must influence ONLY that agent's
    advantages, never a different agent's, within the same env."""
    t, e, n = 3, 1, 2
    rewards = torch.zeros(t, e, n)
    rewards[1, 0, 0] = 100.0  # huge reward, agent 0 only, at t=1
    values = torch.zeros(t, e, n)
    dones = torch.zeros(t, e, dtype=torch.bool)
    dones[-1, 0] = True
    last_value = torch.zeros(e, n)

    adv, _ = compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95)
    assert adv[0, 0, 1] == 0.0  # agent 1 unaffected at t=0
    assert adv[1, 0, 1] == 0.0  # agent 1 unaffected at t=1
    assert adv[2, 0, 1] == 0.0  # agent 1 unaffected at t=2
    assert adv[0, 0, 0] != 0.0  # agent 0's earlier advantage DOES reflect the future reward (that's correct GAE propagation within the SAME agent)


def test_gae_no_leakage_across_envs():
    """A unique reward in one env must not affect another env's advantages."""
    t, e, n = 3, 2, 1
    rewards = torch.zeros(t, e, n)
    rewards[1, 0, 0] = 50.0  # env 0 only
    values = torch.zeros(t, e, n)
    dones = torch.zeros(t, e, dtype=torch.bool)
    dones[-1, :] = True
    last_value = torch.zeros(e, n)

    adv, _ = compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95)
    assert torch.all(adv[:, 1, :] == 0.0)  # env 1 completely unaffected


def test_gae_shape_validation_raises():
    import pytest

    rewards = torch.zeros(2, 1, 1)
    bad_values = torch.zeros(2, 1, 2)  # wrong shape
    dones = torch.zeros(2, 1, dtype=torch.bool)
    last_value = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        compute_gae(rewards, bad_values, dones, last_value, 0.99, 0.95)


def test_normalize_advantages_zero_mean_unit_std():
    adv = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    normed = normalize_advantages(adv)
    assert abs(float(normed.mean())) < 1e-5
    assert abs(float(normed.std(unbiased=False)) - 1.0) < 1e-4


def test_normalize_advantages_excludes_inactive_samples():
    adv = torch.tensor([[[1.0]], [[2.0]], [[1000.0]]])  # last sample is an outlier
    active = torch.tensor([[[True]], [[True]], [[False]]])
    normed = normalize_advantages(adv, active=active)
    # normalization stats computed only from the first two (active) samples
    active_normed = normed[active]
    assert abs(float(active_normed.mean())) < 1e-4

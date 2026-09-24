"""PPO clipped objective, value loss, explained variance tests (§15.7, §33.3)."""

from __future__ import annotations

import torch

from marl.algorithms.common.ppo_update import (
    explained_variance,
    masked_mean,
    ppo_policy_loss,
    value_loss_fn,
)


def test_clip_zero_gradient_outside_clip_range_positive_advantage():
    """With a large positive advantage and ratio far above 1+clip, the
    clipped term dominates and its gradient w.r.t. new_logp should be
    ZERO (the min() picks the clipped, ratio-independent branch)."""
    old_logp = torch.tensor([0.0])
    new_logp = torch.tensor([2.0], requires_grad=True)  # ratio = e^2 ≈ 7.39, way above 1.2
    advantages = torch.tensor([1.0])
    loss, _, _ = ppo_policy_loss(new_logp, old_logp, advantages, clip=0.2)
    loss.backward()
    assert new_logp.grad is not None
    assert abs(float(new_logp.grad)) < 1e-6


def test_clip_nonzero_gradient_inside_clip_range():
    old_logp = torch.tensor([0.0])
    new_logp = torch.tensor([0.01], requires_grad=True)  # ratio ~1.01, inside [0.8,1.2]
    advantages = torch.tensor([1.0])
    loss, _, _ = ppo_policy_loss(new_logp, old_logp, advantages, clip=0.2)
    loss.backward()
    assert abs(float(new_logp.grad)) > 1e-6


def test_clip_fraction_detects_out_of_range_ratios():
    old_logp = torch.zeros(4)
    new_logp = torch.tensor([0.0, 0.01, 5.0, -5.0])  # ratios: 1, ~1.01, huge, ~0
    advantages = torch.ones(4)
    _, _, clip_fraction = ppo_policy_loss(new_logp, old_logp, advantages, clip=0.2)
    assert clip_fraction.item() >= 0.5  # at least the two extreme ones


def test_approx_kl_zero_when_policies_identical():
    old_logp = torch.tensor([0.0, -1.0, -2.0])
    new_logp = old_logp.clone()
    advantages = torch.ones(3)
    _, approx_kl, _ = ppo_policy_loss(new_logp, old_logp, advantages, clip=0.2)
    assert abs(float(approx_kl)) < 1e-6


def test_masked_mean_is_exactly_plain_mean_when_all_active():
    """Bitwise equality with the unmasked path — so v1 (all agents always
    active) is numerically unaffected by agent-masking support."""
    x = torch.randn(257)
    assert torch.equal(masked_mean(x, torch.ones(257, dtype=torch.bool)), x.mean())
    assert torch.equal(masked_mean(x, None), x.mean())


def test_inactive_samples_contribute_no_policy_gradient():
    """§16.6 with an artificial mask: an inactive sample with an enormous
    advantage must not move the policy at all."""
    old_logp = torch.zeros(2)
    new_logp = torch.tensor([0.01, 0.01], requires_grad=True)
    advantages = torch.tensor([1.0, 1e6])
    active = torch.tensor([True, False])
    loss, _, _ = ppo_policy_loss(new_logp, old_logp, advantages, clip=0.2, active=active)
    loss.backward()
    assert new_logp.grad is not None
    assert float(new_logp.grad[1]) == 0.0
    assert float(new_logp.grad[0]) != 0.0


def test_inactive_samples_excluded_from_value_loss():
    values = torch.tensor([1.0, 0.0])
    targets = torch.tensor([1.0, 1000.0])  # huge error on the inactive sample only
    active = torch.tensor([True, False])
    assert float(value_loss_fn(values, targets, active)) == 0.0


def test_value_loss_zero_when_predictions_match_targets():
    values = torch.tensor([1.0, 2.0, 3.0])
    targets = values.clone()
    loss = value_loss_fn(values, targets)
    assert float(loss) < 1e-8


def test_value_loss_matches_half_mse():
    values = torch.tensor([0.0, 0.0])
    targets = torch.tensor([2.0, 2.0])
    loss = value_loss_fn(values, targets)
    assert abs(float(loss) - 0.5 * 4.0) < 1e-6


def test_explained_variance_perfect_prediction():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    returns = values.clone()
    assert abs(explained_variance(values, returns) - 1.0) < 1e-5


def test_explained_variance_zero_variance_returns_nan():
    values = torch.tensor([1.0, 1.0])
    returns = torch.tensor([1.0, 1.0])  # zero variance
    ev = explained_variance(values, returns)
    assert ev != ev  # NaN check (nan != nan)


def test_explained_variance_worse_than_mean_can_be_negative():
    values = torch.tensor([100.0, -100.0, 100.0, -100.0])  # wildly wrong, anti-correlated
    returns = torch.tensor([1.0, 2.0, 1.0, 2.0])  # non-zero variance
    ev = explained_variance(values, returns)
    assert ev < 0

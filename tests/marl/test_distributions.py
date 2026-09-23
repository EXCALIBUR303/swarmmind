"""MaskedCategorical tests (§15.3, §16.8 item 9)."""

from __future__ import annotations

import torch

from marl.algorithms.common.distributions import MaskedCategorical


def test_masked_action_has_zero_probability():
    logits = torch.zeros(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[:, 0] = False
    mask[:, 2] = False
    dist = MaskedCategorical(logits, mask)
    assert torch.allclose(dist.probs[:, 0], torch.zeros(3))
    assert torch.allclose(dist.probs[:, 2], torch.zeros(3))
    assert torch.all(dist.probs[:, [1, 3, 4]] > 0)


def test_probabilities_renormalize_over_valid_actions():
    logits = torch.zeros(1, 5)
    mask = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool)
    dist = MaskedCategorical(logits, mask)
    assert torch.allclose(dist.probs.sum(dim=-1), torch.ones(1))
    assert torch.allclose(dist.probs[0, 0], dist.probs[0, 2])  # equal logits -> equal prob


def test_entropy_is_finite_not_nan():
    logits = torch.randn(4, 5) * 100  # extreme logits, would break naive masking
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[:, 0] = False
    dist = MaskedCategorical(logits, mask)
    entropy = dist.entropy()
    assert torch.all(torch.isfinite(entropy))
    assert torch.all(entropy >= 0)


def test_sample_never_returns_masked_action():
    logits = torch.zeros(1, 5)
    mask = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.bool)
    dist = MaskedCategorical(logits, mask)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        a = dist.sample(gen)
        assert a.item() == 1


def test_sample_reproducible_given_generator():
    logits = torch.randn(1, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    dist = MaskedCategorical(logits, mask)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a1 = [dist.sample(g1).item() for _ in range(20)]
    a2 = [dist.sample(g2).item() for _ in range(20)]
    assert a1 == a2


def test_mode_is_argmax_and_ignores_masked_logits():
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)
    dist = MaskedCategorical(logits, mask)
    assert dist.mode().item() != 0  # action 0 has the highest RAW logit but is masked


def test_log_prob_matches_manual_calculation():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])  # uniform over all 5
    mask = torch.ones(1, 5, dtype=torch.bool)
    dist = MaskedCategorical(logits, mask)
    lp = dist.log_prob(torch.tensor([2]))
    assert torch.allclose(lp, torch.log(torch.tensor(0.2)), atol=1e-5)

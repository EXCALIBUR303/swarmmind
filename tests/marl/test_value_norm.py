"""ValueNorm update/normalize/denormalize round-trip tests (§16.4, §33.3)."""

from __future__ import annotations

import torch

from marl.algorithms.common.value_norm import ValueNorm


def test_normalize_denormalize_round_trip():
    vn = ValueNorm(beta=0.99)
    values = torch.randn(100) * 5 + 10
    for _ in range(200):  # let the running stats converge
        vn.update(values)
    normed = vn.normalize(values)
    denormed = vn.denormalize(normed)
    assert torch.allclose(denormed, values, atol=1e-3)


def test_normalized_values_have_roughly_zero_mean_unit_std_after_convergence():
    vn = ValueNorm(beta=0.99)
    torch.manual_seed(0)
    values = torch.randn(1000) * 3 + 7
    for _ in range(300):
        vn.update(values)
    normed = vn.normalize(values)
    assert abs(float(normed.mean())) < 0.1
    assert abs(float(normed.std()) - 1.0) < 0.1


def test_update_is_running_not_overwriting():
    vn = ValueNorm(beta=0.9)
    vn.update(torch.tensor([0.0]))
    mean_after_one, _ = vn._debiased_mean_var()
    vn.update(torch.tensor([100.0]))
    mean_after_two, _ = vn._debiased_mean_var()
    assert mean_after_two > mean_after_one  # moved toward the new batch, not reset to it
    assert mean_after_two < 100.0  # but not fully overwritten either


def test_state_dict_round_trip():
    vn1 = ValueNorm()
    for _ in range(50):
        vn1.update(torch.randn(10) * 4 + 2)
    state = vn1.state_dict()

    vn2 = ValueNorm()
    vn2.load_state_dict(state)

    test_values = torch.randn(5)
    assert torch.allclose(vn1.normalize(test_values), vn2.normalize(test_values))


def test_variance_floor_prevents_division_explosion():
    """Before any update, variance should be floored (not zero), so
    normalize() never divides by ~0."""
    vn = ValueNorm()
    result = vn.normalize(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.all(torch.isfinite(result))

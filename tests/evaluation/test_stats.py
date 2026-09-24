"""Study statistics vs. scipy reference values (§33.4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from marl.evaluation.stats import (
    hedges_g,
    interp_to_grid,
    normalized_auc,
    t_interval,
    welch_difference,
)

A = [0.82, 0.91, 0.77, 0.88, 0.95]
B = [0.70, 0.74, 0.81, 0.66, 0.79]


def test_t_interval_matches_scipy():
    iv = t_interval(A)
    lo, hi = sps.t.interval(0.95, df=len(A) - 1, loc=np.mean(A), scale=sps.sem(A))
    assert iv.mean == pytest.approx(np.mean(A))
    assert iv.std == pytest.approx(np.std(A, ddof=1))
    assert iv.lo == pytest.approx(lo)
    assert iv.hi == pytest.approx(hi)
    assert iv.n == 5


def test_t_interval_single_seed_has_no_ci():
    iv = t_interval([0.5])
    assert iv.mean == 0.5 and iv.lo is None and iv.hi is None and iv.std is None


def test_t_interval_rejects_nan_and_empty():
    with pytest.raises(ValueError):
        t_interval([0.5, float("nan")])
    with pytest.raises(ValueError):
        t_interval([])


def test_welch_matches_scipy_ttest_and_manual_ci():
    d = welch_difference(A, B)
    ref = sps.ttest_ind(A, B, equal_var=False)
    assert d.diff == pytest.approx(np.mean(A) - np.mean(B))
    assert d.p_value == pytest.approx(ref.pvalue)
    va, vb = np.var(A, ddof=1) / 5, np.var(B, ddof=1) / 5
    df = (va + vb) ** 2 / (va**2 / 4 + vb**2 / 4)
    assert d.df == pytest.approx(df)
    half = sps.t.ppf(0.975, df) * math.sqrt(va + vb)
    assert d.lo == pytest.approx(d.diff - half)
    assert d.hi == pytest.approx(d.diff + half)
    # scipy >= 1.11 exposes the Welch CI directly — cross-check it too.
    ci = ref.confidence_interval(0.95)
    assert d.lo == pytest.approx(ci.low)
    assert d.hi == pytest.approx(ci.high)
    assert d.excludes_zero


def test_welch_unequal_sizes_matches_scipy():
    a, b = [1.0, 2.0, 4.0, 7.0], [3.0, 3.5, 2.5, 4.0, 3.0, 2.0]
    d = welch_difference(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert d.p_value == pytest.approx(ref.pvalue)
    assert (d.lo, d.hi) == (pytest.approx(ci.low), pytest.approx(ci.high))
    assert not d.excludes_zero


def test_welch_degenerate_zero_variance_reports_no_ci():
    d = welch_difference([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert d.diff == 0.0 and d.lo is None and d.p_value is None and not d.excludes_zero
    d2 = welch_difference([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    assert d2.diff == 0.5 and d2.lo is None and not d2.excludes_zero


def test_hedges_g_matches_formula():
    nx, ny = len(A), len(B)
    sp = math.sqrt(((nx - 1) * np.var(A, ddof=1) + (ny - 1) * np.var(B, ddof=1)) / (nx + ny - 2))
    d = (np.mean(A) - np.mean(B)) / sp
    j = 1 - 3 / (4 * (nx + ny) - 9)
    assert hedges_g(A, B) == pytest.approx(j * d)
    assert hedges_g(A, B) == pytest.approx(-hedges_g(B, A))
    assert hedges_g([1.0], [2.0, 3.0]) is None
    assert hedges_g([1.0, 1.0], [1.0, 1.0]) is None


def test_interp_to_grid_never_extrapolates():
    out = interp_to_grid([0, 10, 20], [0.0, 0.5, 1.0], [-5, 0, 5, 20, 25])
    assert np.isnan(out[0]) and np.isnan(out[-1])
    np.testing.assert_allclose(out[1:4], [0.0, 0.25, 1.0])


def test_normalized_auc():
    # Linear 0 -> 1 over [0, 100]: area 50, normalized 0.5.
    assert normalized_auc([0, 50, 100], [0.0, 0.5, 1.0], 100) == pytest.approx(0.5)
    # Constant 1 -> 1.
    assert normalized_auc([0, 100], [1.0, 1.0], 100) == pytest.approx(1.0)
    # Points past x_max are cut off at x_max (interpolated).
    assert normalized_auc([0, 200], [0.0, 1.0], 100) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        normalized_auc([10, 100], [0.0, 1.0], 100)  # does not start at 0
    with pytest.raises(ValueError):
        normalized_auc([0, 50], [0.0, 1.0], 100)  # does not reach x_max

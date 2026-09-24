"""Study statistics (§22.2, §4): the unit of analysis is the training seed.

Every function here takes ONE NUMBER PER SEED (the mean of a metric over
that seed's evaluation episodes) — never per-episode samples, which would
be pseudo-replication (§4). With n=5 seeds the power is low: a CI that
includes 0 is "no detectable difference", never evidence of equivalence.

Reference values are checked against scipy in `tests/evaluation/test_stats.py`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Interval:
    """Mean, sample std (ddof=1) and a two-sided t-interval over seeds."""

    n: int
    mean: float
    std: float | None  # None when n < 2
    lo: float | None  # None when n < 2 (a CI from one seed is undefined)
    hi: float | None
    conf: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Difference:
    """`mean(a) - mean(b)` with a Welch t-interval, Welch's test p-value
    and Hedges' g (bias-corrected standardized mean difference)."""

    n_a: int
    n_b: int
    diff: float
    lo: float | None
    hi: float | None
    df: float | None
    p_value: float | None
    hedges_g: float | None
    conf: float

    @property
    def excludes_zero(self) -> bool:
        return self.lo is not None and self.hi is not None and (self.lo > 0 or self.hi < 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci_excludes_zero"] = self.excludes_zero
        return d


def _clean(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D sequence of per-seed values")
    if np.any(~np.isfinite(arr)):
        raise ValueError("per-seed values must be finite (drop undefined seeds explicitly first)")
    return arr


def t_interval(x: Sequence[float], conf: float = 0.95) -> Interval:
    """Two-sided t confidence interval for the mean of per-seed values."""
    a = _clean(x)
    n = len(a)
    if n == 0:
        raise ValueError("t_interval needs at least one value")
    mean = float(a.mean())
    if n < 2:
        return Interval(n=n, mean=mean, std=None, lo=None, hi=None, conf=conf)
    std = float(a.std(ddof=1))
    half = float(sps.t.ppf(0.5 + conf / 2, df=n - 1)) * std / math.sqrt(n)
    return Interval(n=n, mean=mean, std=std, lo=mean - half, hi=mean + half, conf=conf)


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Hedges' g = J * (mean_a - mean_b) / s_pooled, J = 1 - 3/(4(n_a+n_b)-9).

    None when it is undefined (fewer than 2 values in a group, or zero
    pooled variance)."""
    x, y = _clean(a), _clean(b)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if sp2 <= 0:
        return None
    d = (x.mean() - y.mean()) / math.sqrt(sp2)
    j = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0)
    return float(j * d)


def welch_difference(a: Sequence[float], b: Sequence[float], conf: float = 0.95) -> Difference:
    """Welch (unequal-variance) t-interval for mean(a) - mean(b).

    Degenerate case: if both groups have zero variance (e.g. every seed of
    both methods scores 1.00), the standard error is 0 and the Welch df is
    undefined. The CI is then reported as None, not as a zero-width
    interval — "every seed identical" is not a basis for a significance
    claim with n=5, and the summary says so explicitly."""
    x, y = _clean(a), _clean(b)
    nx, ny = len(x), len(y)
    diff = float(x.mean() - y.mean()) if nx and ny else float("nan")
    g = hedges_g(x.tolist(), y.tolist())
    if nx < 2 or ny < 2:
        return Difference(nx, ny, diff, None, None, None, None, g, conf)
    vx, vy = x.var(ddof=1) / nx, y.var(ddof=1) / ny
    se2 = vx + vy
    if se2 <= 0:
        return Difference(nx, ny, diff, None, None, None, None, g, conf)
    se = math.sqrt(se2)
    df = se2**2 / (vx**2 / (nx - 1) + vy**2 / (ny - 1))
    half = float(sps.t.ppf(0.5 + conf / 2, df=df)) * se
    t_stat = diff / se
    p = float(2.0 * sps.t.sf(abs(t_stat), df=df))
    return Difference(nx, ny, diff, diff - half, diff + half, float(df), p, g, conf)


def interp_to_grid(xs: Sequence[float], ys: Sequence[float], grid: Sequence[float]) -> np.ndarray:
    """Linearly interpolate a learning curve onto a common env-step grid.

    Grid points outside the observed x-range are NaN (never extrapolated)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    g = np.asarray(grid, dtype=float)
    if len(x) == 0:
        return np.full(len(g), np.nan)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    out = np.interp(g, x, y)
    out[(g < x[0]) | (g > x[-1])] = np.nan
    return out


def normalized_auc(xs: Sequence[float], ys: Sequence[float], x_max: float) -> float:
    """Area under a curve on [0, x_max], divided by x_max (§22.2 sample
    efficiency). For a success-rate curve the result is in [0, 1].

    The curve must start at x=0 (the 0% checkpoint) and reach x_max;
    otherwise the area would silently cover a different range per run."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2:
        raise ValueError("normalized_auc needs at least two points")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if x[0] != 0.0 or x[-1] < x_max:
        raise ValueError(f"curve must span [0, {x_max}], got [{x[0]}, {x[-1]}]")
    keep = x <= x_max
    xk, yk = x[keep], y[keep]
    if xk[-1] < x_max:
        xk = np.append(xk, x_max)
        yk = np.append(yk, np.interp(x_max, x, y))
    return float(np.trapz(yk, xk) / x_max)

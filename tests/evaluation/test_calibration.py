"""Calibration gate regression tests (§20.6, Phase 3 deliverable).

Marked `slow` because they run the heuristic/random baselines over full
50-episode validation sets for every level — cheap in absolute terms
(seconds), but skipped from the default fast unit run per the project's
`pytest -m "not slow"` convention (§33.1), and re-run explicitly via
`pytest -m slow` or `make test-slow`.
"""

from __future__ import annotations

import pytest

from scripts.evaluate import evaluate

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("level", ["L1", "L2", "L3", "L4"])
def test_heuristic_meets_calibration_gate(level: str):
    """§20.6: heuristic success on validation maps MUST be >= 0.95 within
    the level's (now-calibrated) max_steps."""
    result = evaluate(level, "heuristic", "validation")
    assert result["success_rate"] >= 0.95, (
        f"{level}: heuristic success {result['success_rate']:.3f} < 0.95 gate"
    )


def test_random_below_threshold_on_l2():
    """§20.6: random success on L2 must stay below 0.6, or the level is
    too easy to demonstrate learning."""
    result = evaluate("L2", "random", "validation")
    assert result["success_rate"] < 0.6, (
        f"L2: random success {result['success_rate']:.3f} >= 0.6 — level may be too easy"
    )


def test_l1_random_baseline_for_phase4_gate_documented():
    """This doesn't assert a pass/fail threshold — it exists so that if the
    L1 random baseline's numbers ever drift (e.g. from an environment
    change), the Phase 4 learning-gate thresholds derived from them in
    docs/EXPERIMENTS.md are flagged as stale rather than silently wrong."""
    result = evaluate("L1", "random", "validation")
    documented_success = 0.58
    documented_censored_steps = 71.48
    assert abs(result["success_rate"] - documented_success) < 0.05, (
        "L1 random success drifted from the value docs/EXPERIMENTS.md's "
        "Phase 4 gate thresholds were derived from — recompute and update both"
    )
    assert abs(result["censored_mean_steps"] - documented_censored_steps) < 5.0, (
        "L1 random censored-mean-steps drifted from the value docs/EXPERIMENTS.md's "
        "Phase 4 gate thresholds were derived from — recompute and update both"
    )

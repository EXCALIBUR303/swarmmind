"""Aggregation on a synthetic study directory (§33.4): every number in the
summary must be recomputable from the per-seed inputs we wrote."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marl.analysis.aggregate import aggregate_study, read_val_curve
from marl.analysis.figures import make_study_figures
from marl.analysis.report import render_results_md
from marl.evaluation.stats import t_interval, welch_difference

SUCCESS = {"ippo": [0.80, 0.85, 0.90], "mappo": [0.95, 0.97, 0.99]}
STRUCT = {"ippo": [0.40, 0.51, 0.45], "mappo": [0.50, 0.49, 0.66]}
RANDOM = [0.50, 0.55, 0.52]


def _agg(success: float, steps: float) -> dict:
    return {
        "n_episodes": 200, "success_rate": success, "censored_steps": steps, "steps_on_success": steps * 0.8,
        "coverage": 0.9, "return_mean": 3.0, "rescue_rate": success, "collisions_per_100_agent_steps": 1.0,
        "exploration_efficiency": 0.1, "idle_fraction": 0.2, "oscillation_rate": 0.05, "workload_gini": 0.1,
    }


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _cfg(smoke: bool = False) -> dict:
    return {"study": "synth", "smoke": smoke, "levels": ["L2"], "algos": ["ippo", "mappo"], "seeds": [0, 1, 2],
            "total_env_steps": 1000, "final_eval": {"map_sets": ["test", "structured"], "episodes_per_map": 2}}


def _build(tmp_path: Path, missing_seed: int | None = None) -> tuple[Path, list[dict]]:
    study_dir = tmp_path / "studies" / "synth"
    runs = []
    for algo in ("ippo", "mappo"):
        for seed in range(3):
            rd = tmp_path / "runs" / algo / f"s{seed}"
            if (algo, seed) == ("mappo", missing_seed):
                _write(rd / "manifest.json", {"status": "interrupted"})
                runs.append({"algo": algo, "level": "L2", "seed": seed, "run_dir": str(rd)})
                continue
            _write(rd / "manifest.json", {"status": "completed"})
            for split, table in (("test", SUCCESS), ("structured", STRUCT)):
                s = table[algo][seed]
                _write(rd / "eval" / f"{split}_L2.json", {"modes": {"stochastic": {"aggregate": _agg(s, 100 - 50 * s)},
                                                                    "greedy": {"aggregate": _agg(s - 0.1, 100)}}})
            # Validation curve with a DUPLICATE update (resume artefact): last one must win.
            lines = [{"update": -1, "env_steps": 0, "success_rate": 0.5},
                     {"update": 4, "env_steps": 500, "success_rate": 0.1},
                     {"update": 4, "env_steps": 500, "success_rate": 0.7},
                     {"update": 9, "env_steps": 1000, "success_rate": SUCCESS[algo][seed]}]
            (rd / "eval_val.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
            runs.append({"algo": algo, "level": "L2", "seed": seed, "run_dir": str(rd)})
    for i, s in enumerate(RANDOM):
        for split in ("validation", "test", "structured"):
            _write(study_dir / "baselines" / "random" / "L2" / f"s{i}" / f"{split}.json",
                   {"modes": {"stochastic": {"aggregate": _agg(s if split != "structured" else s / 5, 120)}}})
    for split in ("validation", "test", "structured"):
        _write(study_dir / "baselines" / "heuristic" / "L2" / "ref" / f"{split}.json",
               {"modes": {"stochastic": {"aggregate": _agg(1.0, 25)}}})
    return study_dir, runs


def test_val_curve_dedupes_resumed_updates(tmp_path):
    _, runs = _build(tmp_path)
    x, y = read_val_curve(Path(runs[0]["run_dir"]))
    assert x == [0.0, 500.0, 1000.0] and y[1] == 0.7


def test_summary_numbers_are_recomputable(tmp_path):
    study_dir, runs = _build(tmp_path)
    s = aggregate_study(study_dir, runs, _cfg())
    lvl = s["levels"]["L2"]
    ippo = lvl["methods"]["ippo"]["splits"]["test"]["stochastic"]["success_rate"]
    ref = t_interval(SUCCESS["ippo"])
    assert ippo["per_seed"] == SUCCESS["ippo"]
    assert ippo["interval"]["lo"] == pytest.approx(ref.lo) and ippo["interval"]["hi"] == pytest.approx(ref.hi)
    d = lvl["comparisons"]["test/stochastic"]["mappo-ippo"]["success_rate"]
    ref_d = welch_difference(SUCCESS["mappo"], SUCCESS["ippo"])
    assert d["diff"] == pytest.approx(ref_d.diff) and d["lo"] == pytest.approx(ref_d.lo)
    # Random only exists in the stochastic mode, so no learned-vs-Random greedy pairs.
    assert set(lvl["comparisons"]["test/greedy"]) == {"mappo-ippo"}
    assert s["missing"] == []


def test_hypothesis_verdicts_follow_the_decision_rules(tmp_path):
    study_dir, runs = _build(tmp_path)
    hyp = aggregate_study(study_dir, runs, _cfg())["levels"]["L2"]["hypotheses"]
    assert hyp["H1"]["ippo"]["verdict"] == "supported" and hyp["H1"]["mappo"]["verdict"] == "supported"
    assert welch_difference(SUCCESS["mappo"], SUCCESS["ippo"]).excludes_zero
    assert hyp["H2"]["verdict"] == "supported_mappo_higher"
    assert "superior" not in json.dumps(hyp).lower()  # §4.2 wording guard
    assert hyp["H3"]["verdict"] == "pending"
    ret = hyp["H4"]["retention"]["ippo"]["per_seed"]
    np.testing.assert_allclose(ret, np.array(STRUCT["ippo"]) / np.array(SUCCESS["ippo"]))


def test_null_result_is_worded_as_no_detectable_difference(tmp_path):
    study_dir, runs = _build(tmp_path)
    # Make MAPPO indistinguishable from IPPO.
    for r in runs:
        if r["algo"] == "mappo":
            p = Path(r["run_dir"]) / "eval" / "test_L2.json"
            obj = json.loads(p.read_text())
            obj["modes"]["stochastic"]["aggregate"]["success_rate"] = SUCCESS["ippo"][r["seed"]] + 0.01
            p.write_text(json.dumps(obj))
    h2 = aggregate_study(study_dir, runs, _cfg())["levels"]["L2"]["hypotheses"]["H2"]
    assert h2["verdict"] == "not_supported"
    assert "not evidence of equivalence" in h2["statement"]


def test_missing_runs_are_reported_and_excluded(tmp_path):
    study_dir, runs = _build(tmp_path, missing_seed=2)
    s = aggregate_study(study_dir, runs, _cfg())
    assert any("mappo/L2/s2" in m for m in s["missing"])
    blk = s["levels"]["L2"]["methods"]["mappo"]["splits"]["test"]["stochastic"]
    assert blk["n_seeds"] == 2


def test_smoke_prefix_figures_and_report(tmp_path):
    study_dir, runs = _build(tmp_path)
    s = aggregate_study(study_dir, runs, _cfg(smoke=True))
    assert all(h["statement"].startswith("[SMOKE") for h in s["levels"]["L2"]["hypotheses"]["H1"].values())
    figs = make_study_figures(s, tmp_path / "figs")
    assert len(figs) == 5 and all(f.stat().st_size > 1000 for f in figs)
    md_path = tmp_path / "docs" / "RESULTS.md"
    md = render_results_md(s, figs, md_path)
    assert "SMOKE STUDY" in md and "H2" in md and "figs/L2_validation_curves.png" in md
    auc = s["levels"]["L2"]["methods"]["ippo"]["validation_curve"]["auc_per_seed"][0]
    # trapezoid of (0,.5),(500,.7),(1000,.8) over 1000
    assert auc == pytest.approx(((0.5 + 0.7) / 2 * 500 + (0.7 + 0.8) / 2 * 500) / 1000)


def test_h2_metric_and_auc_window_follow_the_analysis_config(tmp_path):
    study_dir, runs = _build(tmp_path)
    cfg = {**_cfg(), "analysis": {"h2_primary_metric": "censored_steps", "auc_window_env_steps": 500}}
    lvl = aggregate_study(study_dir, runs, cfg)["levels"]["L2"]
    h2 = lvl["hypotheses"]["H2"]
    steps = {a: [100 - 50 * s for s in SUCCESS[a]] for a in SUCCESS}
    ref = welch_difference(steps["mappo"], steps["ippo"])
    assert h2["metric"] == "censored_steps" and h2["difference"]["diff"] == pytest.approx(ref.diff)
    assert ref.excludes_zero and h2["verdict"] == "supported_mappo_lower" and "(better)" in h2["statement"]
    assert "success_rate" in h2["secondary"]
    auc = lvl["methods"]["ippo"]["validation_curve"]["auc_per_seed"][0]
    assert auc == pytest.approx((0.5 + 0.7) / 2)  # window [0, 500] only

"""Study aggregation (Phase 6 task 3): run directories → `summary.json`.

Unit of analysis = one seed (§4, §22.2). Each learned run contributes the
mean of each metric over its held-out episodes; Random contributes one
number per evaluation sampling seed. Hypothesis statements are generated
here, mechanically, from the pre-registered decision rules — so their
wording can't drift from the numbers (§4.2).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from marl.evaluation.stats import (
    Difference,
    interp_to_grid,
    normalized_auc,
    t_interval,
    welch_difference,
)

LEARNED = ("ippo", "mappo")
LABELS = {"ippo": "IPPO", "mappo": "MAPPO", "random": "Random", "heuristic": "Heuristic"}
# Per-seed test metrics carried into the summary (§23). "higher" says which
# direction is better, used only for wording, never for the decision rule.
METRICS: dict[str, str] = {
    "success_rate": "higher",
    "censored_steps": "lower",
    "steps_on_success": "lower",
    "coverage": "higher",
    "return_mean": "higher",
    "rescue_rate": "higher",
    "collisions_per_100_agent_steps": "lower",
    "exploration_efficiency": "higher",
    "idle_fraction": "lower",
    "oscillation_rate": "lower",
    "workload_gini": "lower",
}
PRIMARY_METRIC = "success_rate"
PRIMARY_SPLIT, PRIMARY_MODE = "test", "stochastic"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_val_curve(run_dir: Path) -> tuple[list[float], list[float]]:
    """Validation success curve from `eval_val.jsonl`. A resumed run can log
    the update it was interrupted in twice; the LAST record per update wins."""
    path = run_dir / "eval_val.jsonl"
    by_update: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                by_update[int(row["update"])] = row
    rows = sorted(by_update.values(), key=lambda r: r["env_steps"])
    return [float(r["env_steps"]) for r in rows], [float(r["success_rate"]) for r in rows]


def _interval_or_none(values: Sequence[float]) -> dict[str, Any] | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return t_interval(vals).to_dict() if vals else None


def _per_seed_block(evals: list[dict[str, Any]], split: str, mode: str) -> dict[str, Any] | None:
    """{metric: {"per_seed": [...], "interval": {...}}} for one split/mode."""
    aggs = [e["modes"][mode]["aggregate"] for e in evals if mode in e.get("modes", {})]
    if not aggs:
        return None
    block: dict[str, Any] = {"n_seeds": len(aggs), "n_episodes_per_seed": [a["n_episodes"] for a in aggs]}
    for metric in METRICS:
        per_seed = [a.get(metric) for a in aggs]
        block[metric] = {"per_seed": per_seed, "interval": _interval_or_none(per_seed)}
    return block


def _values(block: dict[str, Any] | None, metric: str) -> list[float]:
    if not block or metric not in block:
        return []
    return [v for v in block[metric]["per_seed"] if v is not None and np.isfinite(v)]


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}"


def _ci(d: Difference) -> str:
    return "undefined (zero variance in both groups)" if d.lo is None else f"[{d.lo:+.3f}, {d.hi:+.3f}]"


def _h1(algo: str, learned: list[float], random: list[float], prefix: str) -> dict[str, Any]:
    d = welch_difference(learned, random)
    name = LABELS[algo]
    if len(learned) < 2 or len(random) < 2:
        verdict, text = "insufficient", f"{name}: fewer than 2 seeds on one side; H1 cannot be assessed."
    elif d.lo is not None and d.lo > 0:
        verdict = "supported"
        text = f"H1 supported for {name}: test success exceeds Random by {_fmt(d.diff)} (95% Welch CI {_ci(d)})."
    elif d.hi is not None and d.hi < 0:
        verdict = "not_supported"
        text = f"H1 not supported for {name}: its test success is BELOW Random's by {abs(d.diff):.3f} (95% CI {_ci(d)})."
    else:
        verdict = "not_supported"
        text = f"H1 not supported for {name}: the 95% CI of ({name} − Random) {_ci(d)} includes 0."
    return {"verdict": verdict, "difference": d.to_dict(), "statement": prefix + text,
            "n_learned": len(learned), "n_random": len(random)}


METRIC_NAMES = {"success_rate": "test success", "censored_steps": "censored steps-to-completion"}


def _h2(mappo: list[float], ippo: list[float], prefix: str, metric: str = PRIMARY_METRIC) -> dict[str, Any]:
    d = welch_difference(mappo, ippo)
    g = "n/a" if d.hedges_g is None else f"{d.hedges_g:+.2f}"
    what = METRIC_NAMES.get(metric, metric)
    if len(mappo) < 2 or len(ippo) < 2:
        verdict, text = "insufficient", "Fewer than 2 seeds for one algorithm; H2 cannot be assessed."
    elif d.excludes_zero:
        direction = "higher" if d.diff > 0 else "lower"
        better = (direction == "higher") == (METRICS[metric] == "higher")
        verdict = f"supported_mappo_{direction}"
        text = (f"H2 supported: MAPPO's {what} is {direction} ({'better' if better else 'worse'}) than IPPO's by "
                f"{abs(d.diff):.3f} (95% Welch CI of MAPPO − IPPO {_ci(d)}; Hedges' g {g}).")
    else:
        verdict = "not_supported"
        text = (f"H2 not supported: no detectable difference between MAPPO and IPPO in {what} "
                f"(MAPPO − IPPO {_fmt(d.diff)}, 95% Welch CI {_ci(d)}; Hedges' g {g}). With "
                f"{len(mappo)} vs {len(ippo)} seeds the power is low, so this is not evidence of equivalence.")
    return {"verdict": verdict, "metric": metric, "difference": d.to_dict(), "statement": prefix + text}


def _retention(methods: dict[str, Any], algo: str) -> dict[str, Any] | None:
    """H4 (descriptive): per-seed success(structured) / success(test)."""
    m = methods.get(algo, {})
    test = m.get("splits", {}).get("test", {}).get(PRIMARY_MODE)
    struct = m.get("splits", {}).get("structured", {}).get(PRIMARY_MODE)
    if not test or not struct:
        return None
    t, s = test[PRIMARY_METRIC]["per_seed"], struct[PRIMARY_METRIC]["per_seed"]
    ratios = [si / ti for ti, si in zip(t, s, strict=True) if ti is not None and si is not None and ti > 0]
    dropped = len(t) - len(ratios)
    return {"per_seed": ratios, "interval": _interval_or_none(ratios), "seeds_dropped_zero_test_success": dropped}


def _h4(methods: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    ret: dict[str, Any] = {}
    for algo in (*LEARNED, "random"):
        r = _retention(methods, algo)
        if r is not None and r["interval"] is not None:
            ret[algo] = r
    if not ret:
        return None
    parts = []
    for algo, r in ret.items():
        iv = r["interval"]
        ci = f" (95% CI [{iv['lo']:.2f}, {iv['hi']:.2f}])" if iv["lo"] is not None else ""
        parts.append(f"{LABELS[algo]} keeps {iv['mean']:.2f} of its test success on the structured maps{ci}")
    return {"verdict": "descriptive", "retention": ret,
            "statement": prefix + "H4 (descriptive, no threshold): " + "; ".join(parts) + "."}


COORD_ALWAYS = ("der", "visitation_overlap", "mean_inter_robot_distance", "min_inter_robot_distance")
COORD_SOMETIMES = ("der_at_50", "der_iso_at_50", "redundant_pursuit", "allocation_optimality", "dispersion_speed")


def _coord_values(entries: list[dict[str, Any]], field: str) -> list[float]:
    """Per-seed values for one coordination field (§24), pulling `.mean`
    out of the `_SOMETIMES` fields' `{mean, n_defined, n_excluded}` shape
    and skipping any seed where it is undefined."""
    out = []
    for e in entries:
        v = e.get(field)
        if isinstance(v, dict):
            v = v.get("mean")
        if v is not None:
            out.append(float(v))
    return out


# Materiality thresholds for "the ablation degrades coordination-relevant
# metrics" (§24.4 criterion 4). A median collision-rate multiplier > 2x
# from zeroing the teammate channel, or a median success-rate drop > 2
# points from zeroing the visited-known channel, either counts.
ABLATION_COLLISION_MULTIPLIER_THRESHOLD = 2.0
ABLATION_SUCCESS_DROP_THRESHOLD = 0.02


def _ablation_degrades(entry: dict[str, Any]) -> bool:
    mult = entry.get("channel6_collision_multiplier_median")
    drop = entry.get("channel5_success_drop_median")
    return bool((mult is not None and mult > ABLATION_COLLISION_MULTIPLIER_THRESHOLD)
                or (drop is not None and drop > ABLATION_SUCCESS_DROP_THRESHOLD))


def _h3(coord: dict[str, Any] | None, h1: dict[str, Any], ablation_ok: dict[str, bool], prefix: str) -> dict[str, Any]:
    """§24.4's four-criterion "coordination emerged" decision, mechanically
    from already-computed data (H1, this level's coordination.json, and the
    ablation summary). Every criterion must hold, for a given algorithm,
    across seeds, with CIs, or the report says coordination is inconclusive
    for that algorithm — the exact §24.4 fallback wording."""
    if coord is None:
        return {"verdict": "pending", "statement": prefix + "H3 pending: coordination analysis "
                "(scripts/analyze_study.py, §24) has not been run for this study yet."}
    verdicts: dict[str, Any] = {}
    for algo in LEARNED:
        algo_entries = coord["methods"].get(algo, [])
        rnd_entries = coord["methods"].get("random", [])
        if not algo_entries or not rnd_entries:
            verdicts[algo] = {"criteria": {}, "coordination_emerged": False,
                              "statement": f"{LABELS[algo]}: no coordination data."}
            continue
        c1 = h1.get(algo, {}).get("verdict") == "supported"

        der50_algo = _coord_values(algo_entries, "der_at_50")
        der50_rnd = _coord_values(rnd_entries, "der_at_50")
        d_der50_random = welch_difference(der50_algo, der50_rnd)
        vs_random = bool(d_der50_random.hi is not None and d_der50_random.hi < 0)

        # Criterion 2b compares like with like: the team's DER@50 against the isolation null measured
        # at the same 50% coverage, paired by seed (Phase 15 fix; whole-episode DER_iso is inflated by
        # the much longer solo runs and is kept only as a descriptive column).
        paired = []
        for e in algo_entries:
            team, iso = _coord_values([e], "der_at_50"), _coord_values([e], "der_iso_at_50")
            if team and iso:
                paired.append(team[0] - iso[0])
        iv_paired = t_interval(paired) if len(paired) >= 2 else None
        vs_iso = bool(iv_paired and iv_paired.hi is not None and iv_paired.hi < 0)
        c2 = vs_random and vs_iso

        rp_algo = _coord_values(algo_entries, "redundant_pursuit")
        rp_rnd = _coord_values(rnd_entries, "redundant_pursuit")
        d_rp = welch_difference(rp_algo, rp_rnd)
        c3 = bool(d_rp.hi is not None and d_rp.hi < 0)

        c4 = bool(ablation_ok.get(algo, False))

        all_met = c1 and c2 and c3 and c4
        verdicts[algo] = {
            "coordination_emerged": all_met,
            "criteria": {
                "h1_supported": c1,
                "der50_lower_than_random": vs_random,
                "der50_lower_than_isolation_null": vs_iso,
                "der50_vs_random": d_der50_random.to_dict(),
                "der50_vs_isolation_null_paired": iv_paired.to_dict() if iv_paired else None,
                "redundant_pursuit_lower_than_random": c3,
                "redundant_pursuit_vs_random": d_rp.to_dict(),
                "ablation_degrades_coordination": c4,
            },
        }
    all_emerged = all(v.get("coordination_emerged") for v in verdicts.values()) if verdicts else False
    if all_emerged:
        stmt = ("Coordination/cooperation is supported per §24.4 for " + " and ".join(LABELS[a] for a in LEARNED)
                + ": H1 holds, DER@50 is lower than both Random's and the isolation null's, redundant pursuit is "
                "lower than Random's, and the input ablation degrades coordination-relevant metrics.")
    else:
        stmt = "Differences in behaviour were observed; evidence of learned coordination is inconclusive (§24.4)."
    return {"verdict": "emerged" if all_emerged else "inconclusive", "per_algo": verdicts,
            "statement": prefix + stmt}


def _coordination_block(coord: dict[str, Any]) -> dict[str, Any]:
    """Descriptive per-arm coordination summary for the report (§26),
    separate from the §24.4 pass/fail decision in `_h3`."""
    out: dict[str, Any] = {"protocol": coord.get("protocol", {}), "methods": {}}
    for arm, entries in coord["methods"].items():
        block: dict[str, Any] = {"n_seeds": len(entries)}
        for field in (*COORD_ALWAYS, *COORD_SOMETIMES):
            block[field] = _interval_or_none(_coord_values(entries, field))
        der_iso = [e["der_iso"] for e in entries if e.get("der_iso") is not None]
        block["der_iso"] = _interval_or_none(der_iso) if der_iso else None
        out["methods"][arm] = block
    return out


def parameter_counts(obs_radius: int) -> dict[str, int]:
    """Trainable parameters of the networks being compared (§16.3: report them either way, since the
    MAPPO critic is larger — a capacity confound for any MAPPO advantage)."""
    from marl.envs.core.global_state import N_GLOBAL_CHANNELS, N_GLOBAL_VEC_FEATURES
    from marl.networks.actor import ActorPolicy
    from marl.networks.critics import CentralCritic, LocalCritic

    k = 2 * obs_radius + 1
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    return {
        "actor": count(ActorPolicy(grid_size=k)),
        "ippo_critic": count(LocalCritic(grid_size=k)),
        "mappo_critic": count(CentralCritic(N_GLOBAL_CHANNELS, grid_size=k, global_vec_dim=N_GLOBAL_VEC_FEATURES)),
    }


def aggregate_study(study_dir: Path, runs: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """`runs`: [{"algo","level","seed","run_dir"}] for every planned learned
    job (run_dir None if it never completed)."""
    smoke = bool(cfg.get("smoke"))
    analysis = cfg.get("analysis") or {}
    prefix = "[SMOKE — infrastructure dry run, not evidence] " if smoke else ""
    summary: dict[str, Any] = {
        "study": cfg["study"],
        "smoke": smoke,
        "generated": datetime.now(UTC).isoformat(),
        "unit_of_analysis": "training seed (learned) / evaluation sampling seed (Random)",
        "primary": {"metric": PRIMARY_METRIC, "h2_metric": analysis.get("h2_primary_metric", PRIMARY_METRIC),
                    "auc_window_env_steps": analysis.get("auc_window_env_steps"),
                    "split": PRIMARY_SPLIT, "mode": PRIMARY_MODE, "checkpoint": "final"},
        "config": cfg,
        "levels": {},
        "missing": [],
    }
    for level in cfg["levels"]:
        methods: dict[str, Any] = {}
        grid: list[float] | None = None
        for algo in cfg["algos"]:
            entries = sorted((r for r in runs if r["algo"] == algo and r["level"] == level), key=lambda r: r["seed"])
            done: list[int] = []
            evals: dict[str, list[dict[str, Any]]] = {}
            curves: list[tuple[int, list[float], list[float]]] = []
            for r in entries:
                rd = Path(r["run_dir"]) if r.get("run_dir") else None
                manifest = _read_json(rd / "manifest.json") if rd else None
                if rd is None or not manifest or manifest.get("status") != "completed":
                    summary["missing"].append(f"{algo}/{level}/s{r['seed']}: run not completed")
                    continue
                done.append(r["seed"])
                for split in cfg["final_eval"]["map_sets"]:
                    e = _read_json(rd / "eval" / f"{split}_{level}.json")
                    if e is None:
                        summary["missing"].append(f"{algo}/{level}/s{r['seed']}: no {split} eval")
                    else:
                        evals.setdefault(split, []).append(e)
                curves.append((r["seed"], *read_val_curve(rd)))
            splits = {
                split: {mode: _per_seed_block(es, split, mode) for mode in ("stochastic", "greedy")}
                for split, es in evals.items()
            }
            curve_block: dict[str, Any] = {}
            if curves:
                grid = grid or curves[0][1]
                x_max = min(c[1][-1] for c in curves if c[1])
                if analysis.get("auc_window_env_steps"):
                    x_max = min(x_max, float(analysis["auc_window_env_steps"]))
                ys = np.array([interp_to_grid(x, y, grid) for _, x, y in curves])
                means, los, his = [], [], []
                for j in range(len(grid)):
                    col = ys[:, j][np.isfinite(ys[:, j])]
                    iv = t_interval(col) if len(col) else None
                    means.append(iv.mean if iv else None)
                    los.append(iv.lo if iv else None)
                    his.append(iv.hi if iv else None)
                aucs = [normalized_auc(x, y, x_max) for _, x, y in curves if x and x[0] == 0 and x[-1] >= x_max]
                curve_block = {
                    "grid_env_steps": list(grid),
                    "seeds": [c[0] for c in curves],
                    "per_seed": [[None if not np.isfinite(v) else float(v) for v in row] for row in ys],
                    "mean": means, "lo": los, "hi": his,
                    "auc_x_max": x_max,
                    "auc_per_seed": aucs,
                    "auc": _interval_or_none(aucs),
                    "untrained_val_success": _interval_or_none([y[0] for _, x, y in curves if x and x[0] == 0]),
                    "final_val_success": _interval_or_none([y[-1] for _, _, y in curves if y]),
                }
            methods[algo] = {"seeds_completed": done, "splits": splits, "validation_curve": curve_block}

        for kind in ("random", "heuristic"):
            base = study_dir / "baselines" / kind / level
            tags = sorted(p.name for p in base.iterdir()) if base.is_dir() else []
            block: dict[str, Any] = {"sampling_seeds": tags, "splits": {}}
            for split in ["validation", *cfg["final_eval"]["map_sets"]]:
                es = [e for t in tags if (e := _read_json(base / t / f"{split}.json"))]
                if es:
                    block["splits"][split] = {"stochastic": _per_seed_block(es, split, "stochastic")}
            if tags:
                methods[kind] = block

        comparisons: dict[str, Any] = {}
        for split in cfg["final_eval"]["map_sets"]:
            for mode in ("stochastic", "greedy"):
                blk = {a: methods.get(a, {}).get("splits", {}).get(split, {}).get(mode) for a in LEARNED}
                rnd = methods.get("random", {}).get("splits", {}).get(split, {}).get("stochastic")
                pairs = {"mappo-ippo": (blk["mappo"], blk["ippo"]), "ippo-random": (blk["ippo"], rnd),
                         "mappo-random": (blk["mappo"], rnd)}
                comp = {}
                for name, (a, b) in pairs.items():
                    # Random is evaluated stochastically only, so learned-vs-Random
                    # pairs exist only in the stochastic (primary) mode.
                    if a and b and (mode == "stochastic" or name == "mappo-ippo"):
                        comp[name] = {m: welch_difference(_values(a, m), _values(b, m)).to_dict()
                                      for m in METRICS if len(_values(a, m)) and len(_values(b, m))}
                if comp:
                    comparisons[f"{split}/{mode}"] = comp
        auc = {a: methods.get(a, {}).get("validation_curve", {}).get("auc_per_seed", []) for a in LEARNED}
        if auc["mappo"] and auc["ippo"]:
            comparisons["validation_auc"] = {"mappo-ippo": welch_difference(auc["mappo"], auc["ippo"]).to_dict()}

        hyp: dict[str, Any] = {}
        rnd_blk = methods.get("random", {}).get("splits", {}).get(PRIMARY_SPLIT, {}).get(PRIMARY_MODE)
        for algo in LEARNED:
            lb = methods.get(algo, {}).get("splits", {}).get(PRIMARY_SPLIT, {}).get(PRIMARY_MODE)
            if lb and rnd_blk:
                hyp.setdefault("H1", {})[algo] = _h1(algo, _values(lb, PRIMARY_METRIC), _values(rnd_blk, PRIMARY_METRIC),
                                                    prefix)
        mb = methods.get("mappo", {}).get("splits", {}).get(PRIMARY_SPLIT, {}).get(PRIMARY_MODE)
        ib = methods.get("ippo", {}).get("splits", {}).get(PRIMARY_SPLIT, {}).get(PRIMARY_MODE)
        if mb and ib:
            h2_metric = analysis.get("h2_primary_metric", PRIMARY_METRIC)
            hyp["H2"] = _h2(_values(mb, h2_metric), _values(ib, h2_metric), prefix, h2_metric)
            test_comp = comparisons.get("test/stochastic", {}).get("mappo-ippo", {})
            hyp["H2"]["secondary"] = {
                **{m: test_comp.get(m) for m in ("success_rate", "censored_steps") if m != h2_metric},
                "validation_auc": comparisons.get("validation_auc", {}).get("mappo-ippo"),
            }
        coord = _read_json(study_dir / "analysis" / f"{level}_coordination.json")
        ablation_ok = {
            algo: _ablation_degrades(coord.get("ablation", {}).get(algo, {})) for algo in LEARNED
        } if coord else {}
        hyp["H3"] = _h3(coord, hyp.get("H1", {}), ablation_ok, prefix)
        h4 = _h4(methods, prefix)
        if h4:
            hyp["H4"] = h4
        from marl.envs.core.config import LevelRegistry

        level_summary: dict[str, Any] = {"methods": methods, "comparisons": comparisons, "hypotheses": hyp,
                                         "parameter_counts": parameter_counts(LevelRegistry().get(level).obs_radius)}
        if coord:
            level_summary["coordination"] = _coordination_block(coord)
        summary["levels"][level] = level_summary
    return summary

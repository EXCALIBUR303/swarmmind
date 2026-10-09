"""Real-data audit (Phase 14): can every number the system shows or documents be traced to a
results file, and can those files be traced to the trained models?

Runs entirely against the committed demo bundle (`demo_data/`), installed into a temp
`results/`, so it means the same on a clean clone. Writes `docs/REAL_DATA_AUDIT.md` and exits
non-zero if any row FAILS. Nothing here is a mock: every check re-derives a number from a
more primary artifact and compares.

    .venv/bin/python scripts/audit_real_data.py [--out docs/REAL_DATA_AUDIT.md] [--skip-slow]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from marl.utils.paths import REPO_ROOT

STUDY, LEVEL = "main_L2", "L2"
ORIGINAL_ROOT = "/Users/sid/Claude/swarmmind/results"  # absolute paths baked into the study manifest at training time


@dataclass
class Row:
    id: str
    claim: str
    source: str
    method: str
    status: str  # PASS | FAIL | NOT VERIFIABLE
    detail: str
    seconds: float = 0.0


ROWS: list[Row] = []


def row(id_: str, claim: str, source: str, method: str):
    """Decorator: run `fn() -> (ok: bool | None, detail)` and record the outcome (None = not verifiable)."""

    def deco(fn):
        t0 = time.time()
        try:
            ok, detail = fn()
            status = ok if isinstance(ok, str) else ("NOT VERIFIABLE" if ok is None else ("PASS" if ok else "FAIL"))
        except Exception as e:  # noqa: BLE001 - an audit must report, not crash
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        ROWS.append(Row(id_, claim, source, method, status, detail, time.time() - t0))
        print(f"[{status}] {id_} {claim} ({time.time() - t0:.1f}s) {detail if status == 'FAIL' else detail[:200]}", flush=True)
        return fn

    return deco


def jload(p: Path):
    return json.loads(p.read_text())


def contains(big, small, path="") -> list[str]:
    """Differences where `small` is not a sub-structure of `big` (lists: each small item matches some big item)."""
    if isinstance(small, dict):
        if not isinstance(big, dict):
            return [f"{path}: expected object"]
        out: list[str] = []
        for k, v in small.items():
            out += [f"{path}/{k}: missing"] if k not in big else contains(big[k], v, f"{path}/{k}")
        return out
    if isinstance(small, list):
        if not isinstance(big, list):
            return [f"{path}: expected array"]
        if len(small) == len(big):
            direct = [d for i, (b, s) in enumerate(zip(big, small, strict=True)) for d in contains(b, s, f"{path}/{i}")]
            if not direct:
                return []
        out = []
        for i, s in enumerate(small):
            if not any(not contains(b, s) for b in big):
                out.append(f"{path}/{i}: no matching element in the real data")
        return out
    return [] if big == small else [f"{path}: {small!r} != {big!r}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "REAL_DATA_AUDIT.md")
    ap.add_argument("--skip-slow", action="store_true", help="skip the held-out re-evaluations")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="sm_audit_"))
    res = tmp / "results"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "demo_setup.py"), "--results", str(res)], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        return 2
    studies, runs_root, replays_root = res / "studies", res / STUDY, res / "replays" / STUDY
    runs_root = res / "runs" / STUDY
    run_dirs = sorted(p for p in runs_root.glob("*/*/*") if p.is_dir())

    @row("A1", "The bundle is exactly what the manifest says (133 files, SHA-256 each)", "demo_data/MANIFEST.json",
         "scripts/demo_setup.py --check on a fresh install")
    def _():
        c = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "demo_setup.py"), "--results", str(res), "--check"], capture_output=True, text=True)
        return c.returncode == 0, (c.stdout or c.stderr).strip()

    @row("A2", "All 10 study runs finished, from a clean git tree, and their ids carry their config hash", "runs/*/manifest.json",
         "manifest status, git_dirty, config_hash == id suffix, trained steps")
    def _():
        bad, commits = [], set()
        for d in run_dirs:
            m = jload(d / "manifest.json")
            commits.add(m["versions"]["git_commit"][:8])
            if m["status"] != "completed" or m["versions"]["git_dirty"] or not d.name.endswith(m["config_hash"]) or m["trained_env_steps"] < 0.99 * m["budget_env_steps"]:
                bad.append(d.name)
        return not bad and len(run_dirs) == 10, f"{len(run_dirs)} runs; training commits {sorted(commits)}; offenders {bad}"

    @row("B1", "Every number in summary.json (and so in RESULTS.md and the Compare page) is recomputed from the per-run eval files",
         "studies/main_L2/summary.json <- runs/*/eval/*.json + baselines/",
         "re-run marl.analysis.aggregate.aggregate_study on the bundle, diff every leaf (only `generated` may differ)")
    def _():
        from marl.analysis.aggregate import aggregate_study
        from marl.experiments.config import load_study_config
        from marl.experiments.outputs import planned_runs

        sd = studies / STUDY
        man = sd / "study_manifest.json"
        man.write_text(man.read_text().replace(ORIGINAL_ROOT, str(res)))  # point the manifest at THIS copy, not the original disk
        cfg = load_study_config(STUDY)
        fresh = json.loads(json.dumps(aggregate_study(sd, planned_runs(cfg, studies), cfg.model_dump())))
        ref = jload(sd / "summary.json")

        def walk(a, b, p=""):
            if type(a) is not type(b):
                return [p]
            if isinstance(a, dict):
                return [d for k in set(a) | set(b) for d in (walk(a[k], b[k], f"{p}/{k}") if k in a and k in b else [f"{p}/{k}"])]
            if isinstance(a, list):
                return [p] if len(a) != len(b) else [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in walk(x, y, f"{p}/{i}")]
            return [] if a == b else [p]

        diffs = [d for d in walk(fresh, ref) if d != "/generated"]
        return not diffs, f"{len(diffs)} differing leaves" + (f": {diffs[:5]}" if diffs else " (all leaves identical)")

    @row("B2", "docs/RESULTS.md is summary.json rendered, with nothing hand-written", "docs/RESULTS.md <- summary.json",
         "re-render with marl.analysis.report.render_results_md and diff the text (header timestamp excluded)")
    def _():
        from marl.analysis.figures import make_study_figures
        from marl.analysis.report import render_results_md

        summary = jload(studies / STUDY / "summary.json")
        fake_docs = tmp / "docs"
        figs = make_study_figures(summary, fake_docs / "figures" / STUDY)
        text = render_results_md(summary, figs, fake_docs / "RESULTS.md")
        strip = lambda s: re.sub(r"^\*Generated by.*$", "", s, flags=re.M)  # noqa: E731
        committed = (REPO_ROOT / "docs" / "RESULTS.md").read_text()
        a, b = strip(text).splitlines(), strip(committed).splitlines()
        diff = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y]
        return len(a) == len(b) and not diff, f"{len(b)} lines; {len(diff)} differ" + (f"; first: line {diff[0][0]}" if diff else "")

    if not args.skip_slow:
        @row("C1", "The trained actors produce the stored held-out results (structured maps, all 10 runs, stochastic)",
             "runs/*/actor.pt -> runs/*/eval/structured_L2.json", "re-evaluate each actor with marl.evaluation.final_eval.evaluate_policy; compare every episode record")
        def _():
            return reevaluate(run_dirs, "structured", tmp)

        @row("C2", "...and on the 200-episode test set (one IPPO and one MAPPO run)", "runs/*/actor.pt -> runs/*/eval/test_L2.json",
             "same, split=test")
        def _():
            pick = [next(d for d in run_dirs if d.parts[-3] == algo and d.name.startswith("s0-")) for algo in ("ippo", "mappo")]
            return reevaluate(pick, "test", tmp)

    @row("D1", "Each stored replay's outcome equals the same episode in the run's eval file (final-checkpoint replays)",
         "replays/main_L2/*/*.json.gz vs runs/*/eval/structured_L2.json", "match on (map, eval seed 0): success and steps")
    def _():
        checked = mismatches = 0
        for d in run_dirs:
            rid = d.name
            ev = jload(d / "eval" / "structured_L2.json")["modes"]["stochastic"]["episodes"]
            by = {(e["map_id"], e["eval_seed"]): e for e in ev}
            for f in (replays_root / rid).glob("*step999424_eval0.json.gz") if (replays_root / rid).is_dir() else []:
                with gzip.open(f) as fh:
                    rep = json.load(fh)
                e = by[(rep["map"]["id"], 0)]
                checked += 1
                mismatches += int(e["success"] != rep["outcome"]["success"] or e["steps"] != rep["outcome"]["steps"])
        return checked == 10 and mismatches == 0, f"{checked} final-checkpoint replays checked, {mismatches} mismatches"

    @row("E1", "Frontend test fixtures are real API/file payloads (contained in the live data, not invented)",
         "frontend/src/features/**/__fixtures__/*.json", "structural containment in the bundle's files / API payloads")
    def _():
        return audit_fixtures(res)

    @row("E2", "Production frontend code imports no mock/fixture and contains no hard-coded result number",
         "frontend/src (non-test)", "scripts/check_no_mocks.mjs + scan for 3+-decimal numeric literals")
    def _():
        env = {"NODE_OPTIONS": "--no-network-family-autoselection", "PATH": __import__("os").environ["PATH"]}
        c = subprocess.run(["node", "scripts/check_no_mocks.mjs"], cwd=REPO_ROOT / "frontend", capture_output=True, text=True, env=env)
        hits = []
        for p in (REPO_ROOT / "frontend" / "src").rglob("*.ts*"):
            if re.search(r"(\.test\.|__fixtures__|\.gen\.|fixtures\.ts|test-setup)", str(p)) or p.name == "contrast.ts":
                continue  # contrast.ts: the WCAG 2.x luminance formula's published constants, not data
            for n, line in enumerate(p.read_text().splitlines(), 1):
                code = re.sub(r"//.*$|/\*.*?\*/|`[^`]*`|\"[^\"]*\"|'[^']*'", "", line)  # numbers in strings/comments are not data
                if re.search(r"(?<![\w.#])\d+\.\d{3,}(?![\w])", code):
                    hits.append(f"{p.relative_to(REPO_ROOT)}:{n}: {line.strip()[:90]}")
        return c.returncode == 0 and not hits, f"check_no_mocks exit {c.returncode}; {len(hits)} decimal literals" + (f": {hits[:4]}" if hits else "")

    @row("F1", "Every result number quoted in README.md / docs/ARCHITECTURE.md / docs/DEMO.md is in summary.json",
         "README.md, docs/*.md", "extract numeric claims (x.xx, x%), check each against the rounded leaves of summary.json")
    def _():
        return audit_doc_numbers(jload(studies / STUDY / "summary.json"))

    if not args.skip_slow:
        @row("G1", "What the API serves equals what is on disk, and live sessions equal stored replays (real server, real sockets)",
             "tests/integration/test_stack.py", "pytest -k 'numbers or reproduce or direct_evaluator'")
        def _():
            c = subprocess.run([sys.executable, "-m", "pytest", "tests/integration/test_stack.py", "-q", "-k", "every_api_number or reproduce_every or direct_evaluator"],
                               cwd=REPO_ROOT, capture_output=True, text=True)
            return c.returncode == 0, c.stdout.strip().splitlines()[-1] if c.stdout.strip() else c.stderr[-200:]

    write_report(args.out)
    return 0 if all(x.status != "FAIL" for x in ROWS) else 1


def reevaluate(run_dirs: list[Path], split: str, tmp: Path) -> tuple[bool, str]:
    from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
    from marl.envs.core.config import LevelRegistry
    from marl.evaluation.final_eval import evaluate_policy, load_heldout_maps

    cfg = LevelRegistry().get(LEVEL)
    maps = load_heldout_maps(LEVEL, split, {"requester": "audit_real_data"}, log_path=tmp / "heldout_access_log.jsonl")  # gatekeeper, private log
    total = bad = 0
    for d in run_dirs:
        stored = jload(d / "eval" / f"{split}_{LEVEL}.json")["modes"]["stochastic"]["episodes"]
        by = {(e["map_id"], e["eval_seed"]): e for e in stored}
        actor, _ = load_actor(str(d / "actor.pt"))
        fresh = evaluate_policy(cfg, lambda a=actor: LearnedActorPolicy(a, stochastic=True), maps, eval_seeds=[0, 1])["episodes"]
        assert len(fresh) == len(stored), (len(fresh), len(stored))
        for e in fresh:
            ref = by[(e["map_id"], e["eval_seed"])]
            total += 1
            same = all((np.isclose(e[k], ref[k], rtol=0, atol=1e-9) if isinstance(ref[k], float) else e[k] == ref[k]) for k in ref)
            bad += int(not same)
    return bad == 0, f"{len(run_dirs)} actor(s), {total} episodes re-run; {bad} differ from the stored records"


def audit_fixtures(res: Path) -> tuple[bool | str | None, str]:
    """Each frontend test fixture must be a sub-structure of the real file/payload it claims to be."""
    from backend.app.schemas.training import train_schema
    from backend.app.services.study_service import preregistration_rules

    fx = REPO_ROOT / "frontend" / "src" / "features" / "__fixtures__"
    sd = res / "studies" / STUDY
    summary = jload(sd / "summary.json")
    run = jload(fx / "run_detail.json")["summary"]["run_id"]
    rdir = next((res / "runs" / STUDY).glob(f"*/*/{run}"))
    manifests = [jload(p / "manifest.json") for p in sorted((res / "runs" / STUDY).glob("*/*/*"))]
    have_replays = {p.name.removesuffix(".json.gz") for p in (res / "replays" / STUDY).rglob("*.json.gz")}
    problems: list[str] = []
    verified: list[str] = []
    partial: list[str] = []

    def check(name: str, diffs: list[str]) -> None:
        (problems.append(f"{name}: {diffs[:2]}") if diffs else verified.append(name))

    for f in sorted(fx.glob("*.json")):
        data = jload(f)
        n = f.name
        if n.startswith("analysis_"):
            check(n, contains(jload(sd / "analysis" / n.removeprefix("analysis_")), data))
        elif n == "study.json":
            check(n, contains({"summary": summary}, {"summary": data["summary"]}))
        elif n == "run_metrics.json":
            check(n, contains([json.loads(x) for x in (rdir / "metrics.jsonl").read_text().splitlines()], data))
        elif n == "run_detail.json":
            derived = {k: v for k, v in data["manifest"].items() if k != "n_checkpoints"}  # counted by the API from checkpoints/, which the bundle omits
            check(n, contains({"manifest": jload(rdir / "manifest.json")}, {"manifest": derived}))
        elif n == "run_eval.json":
            rows = [json.loads(x) for x in (rdir / "eval_val.jsonl").read_text().splitlines()]
            check(n, contains(rows, data["validation_history"]) + contains(jload(rdir / "eval" / "structured_L2.json"), data["structured_L2"])
                  + contains(jload(rdir / "eval" / "test_L2.json"), data["test_L2"]))
        elif n == "prereg.json":
            check(n, contains(preregistration_rules(), data))
        elif n == "train_schema.json":
            check(n, contains(train_schema(data["total_env_steps"]["max"]), data))
        elif n == "studies.json":
            ent = next(e for e in data if e["study"] == STUDY)
            check(n, contains({**summary, "levels": list(summary["levels"])}, {k: ent[k] for k in ("study", "smoke", "generated", "levels", "primary")}))
            if len(data) > 1:
                partial.append(f"studies.json also lists {[e['study'] for e in data if e['study'] != STUDY]} (not in the bundle)")
        elif n == "runs.json":
            ids = {m["run_id"]: m for m in manifests}
            mine = [r for r in data if r.get("study") == STUDY]
            check(n, [f"{r['run_id']}: not in bundle" for r in mine if r["run_id"] not in ids]
                  + [d for r in mine if r["run_id"] in ids for d in contains(ids[r["run_id"]], {k: r[k] for k in ("algo", "seed", "level", "status") if k in r})])
            if len(mine) != len(data):
                partial.append(f"runs.json lists {len(data) - len(mine)} pilot/gate runs (not in the bundle)")
        elif n == "replays_index.json":
            mine = [e for e in data if e.get("study") == STUDY]
            check(n, [e["replay_id"] for e in mine if e["replay_id"] not in have_replays])
            if len(mine) != len(data):
                partial.append(f"replays_index.json lists {len(data) - len(mine)} replays from other studies (not in the bundle)")
        else:
            problems.append(f"{n}: no verifier written for this fixture")
    sim = REPO_ROOT / "frontend" / "src" / "features" / "simulation" / "__fixtures__"
    for f in sorted(sim.glob("*.json")):
        data = jload(f)
        hit = next((p for p in (res / "replays" / STUDY).rglob("*.json.gz") if p.name == f"{data['replay_id']}.json.gz"), None)
        if hit is None:
            problems.append(f"{f.name}: replay {data['replay_id']} not in bundle")
            continue
        with gzip.open(hit) as fh:
            stored = json.load(fh)
        derived = {k: v for k, v in data.items() if k != "initial"}  # `initial` is derived by the API for pre-Phase-10 replays
        check(f.name, contains(stored, derived))
    detail = f"{len(verified)} fixtures verified as sub-structures of real files/payloads"
    if partial:
        detail += "; limits: " + "; ".join(partial)
    if problems:
        return False, detail + f"; PROBLEMS {problems}"
    return ("PARTIAL" if partial else True), detail


NUM_RE = re.compile(r"(?<![\w.])(\d+\.\d{2,3})(?![\w.+])")  # `20.19+` is a version, not a result


def _doc_claims(summary: dict) -> dict[str, list[tuple[str, float]]]:
    """Every result number the prose may quote -> the specific summary.json value(s) it must round to.
    A number in a doc that is not registered here FAILS the audit (so new unchecked numbers cannot slip in)."""
    L = summary["levels"][LEVEL]
    test = lambda m: L["methods"][m]["splits"]["test"]["stochastic"]["success_rate"]["interval"]["mean"]  # noqa: E731
    H = L["hypotheses"]
    return {
        "0.44": [("Random held-out test success (mean over its 5 sampling seeds)", test("random"))],
        "1.00": [("IPPO held-out test success (mean over 5 seeds)", test("ippo")), ("MAPPO held-out test success (mean over 5 seeds)", test("mappo"))],
        "0.56": [("H1 IPPO: success difference vs Random", H["H1"]["ippo"]["difference"]["diff"])],
        "0.32": [("H2: |MAPPO - IPPO| censored steps difference", abs(H["H2"]["difference"]["diff"]))],
        "0.83": [("H4: IPPO structured/test retention", H["H4"]["retention"]["ippo"]["interval"]["mean"])],
        "0.82": [("H4: MAPPO structured/test retention", H["H4"]["retention"]["mappo"]["interval"]["mean"])],
        "0.33": [("H4: Random structured/test retention", H["H4"]["retention"]["random"]["interval"]["mean"])],
    }


def audit_doc_numbers(summary: dict) -> tuple[bool, str]:
    claims = _doc_claims(summary)
    checked, bad = 0, []
    used: set[str] = set()
    for name in ("README.md", "docs/ARCHITECTURE.md", "docs/DEMO.md"):
        p = REPO_ROOT / name
        if not p.is_file():
            continue
        text = re.sub(r"```.*?```", "", p.read_text(), flags=re.S)
        for m in NUM_RE.finditer(text):
            tok = m.group(1)
            if text[max(0, m.start() - 7):m.start()].lower().endswith(("python ", "node ", "ode.js ")):
                continue  # a tool version, not a result
            checked += 1
            if tok not in claims:
                bad.append(f"{name}: {tok} (not a registered, checked claim)")
                continue
            tol = 0.5 * 10 ** -len(tok.split(".")[1]) + 1e-9
            used.add(tok)
            if not all(abs(v - float(tok)) <= tol for _, v in claims[tok]):
                bad.append(f"{name}: {tok} != {[(d, round(v, 4)) for d, v in claims[tok]]}")
    return not bad, f"{checked} numeric claims, each matched to a named summary.json value ({len(used)} distinct); failures: {bad[:6]}"


def write_report(out: Path) -> None:
    n_fail = sum(r.status == "FAIL" for r in ROWS)
    lines = ["# Real-data audit", "",
             f"Generated by `scripts/audit_real_data.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}. Do not edit by hand. "
             "It runs against the committed demo bundle only (`demo_data/`, installed into a temporary `results/`), so the outcome is the same on a clean clone.", "",
             f"**Overall: {'FAIL (' + str(n_fail) + ' row(s))' if n_fail else 'PASS'}** — "
             f"{sum(r.status == 'PASS' for r in ROWS)} pass, {sum(r.status == 'PARTIAL' for r in ROWS)} partial, {sum(r.status == 'NOT VERIFIABLE' for r in ROWS)} not verifiable, {n_fail} fail.", "",
             "A row PASSES only when a number was **re-derived from a more primary artifact and matched**; anything that could not be checked "
             "is reported as NOT VERIFIABLE, never as PASS.", "",
             "| # | Claim | Traced from → to | Method | Result | Evidence |", "|---|---|---|---|---|---|"]
    for r in ROWS:
        lines.append(f"| {r.id} | {r.claim} | `{r.source}` | {r.method} | **{r.status}** | {r.detail.replace('|', '/')} |")
    lines += ["", "## What this chain establishes", "",
              "```",
              "actor.pt (trained weights)  --C1/C2: re-run held-out episodes-->  runs/*/eval/*.json",
              "runs/*/eval/*.json          --B1: re-aggregate------------------>  studies/main_L2/summary.json",
              "summary.json                --B2: re-render--------------------->  docs/RESULTS.md",
              "summary.json / replays      --G1: served by the API, shown by the UI (E2E: frontend/e2e/run_e2e.py)",
              "```", "",
              "Not covered here (see `docs/E2E_REPORT.md`): the browser run compares what the pages display with these same files."]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())

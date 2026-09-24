#!/usr/bin/env python
"""Run (or continue) a pre-registered study end to end (Phase 6).

Idempotent: re-running the same command skips completed runs, resumes
interrupted ones from their last checkpoint, and only re-evaluates what is
missing. Safe to Ctrl-C and restart.

Usage:
    python scripts/run_study.py --study smoke --parallel 4
    python scripts/run_study.py --study main_L2 --parallel 8          # Phase 7, in a terminal
    python scripts/run_study.py --study main_L2 --plan                # show what would run
    python scripts/run_study.py --study main_L2 --freeze              # pre-register (hash the inputs)
    python scripts/run_study.py --study smoke --stages aggregate      # regenerate RESULTS.md only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.experiments.config import load_study_config, study_config_path
from marl.experiments.outputs import record_evolution_replays, write_study_outputs
from marl.experiments.runner import (
    check_frozen,
    classify_run,
    freeze_study,
    frozen_inputs,
    plan_train_jobs,
    resolved_run_config,
    run_study,
)
from marl.utils.hashing import config_hash
from marl.utils.paths import RUNS_DIR

ALL_STAGES = ("train", "eval", "baselines", "aggregate", "replays")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", required=True, help="study name in configs/experiments/ or a YAML path")
    p.add_argument("--parallel", type=int, default=4, help="worker processes (1 torch thread each); 0 = inline")
    p.add_argument("--stages", default=",".join(ALL_STAGES), help=f"comma-separated subset of {ALL_STAGES}")
    p.add_argument("--plan", action="store_true", help="print each job's state and exit")
    p.add_argument("--freeze", action="store_true", help="record the study's input hashes in FROZEN.json and exit")
    p.add_argument("--allow-deviation", action="store_true", help="run a frozen study whose inputs changed (recorded)")
    p.add_argument("--results-md", default=None, help="where to write RESULTS.md (default docs/RESULTS.md)")
    args = p.parse_args()

    cfg = load_study_config(args.study)
    yaml_path = Path(args.study) if Path(args.study).suffix else study_config_path(cfg.study)
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        p.error(f"unknown stages {sorted(unknown)}")

    if args.freeze:
        entry = freeze_study(cfg, yaml_path)
        print(f"froze study '{cfg.study}' at git {entry['git']}:")
        for path, h in entry["files"].items():
            print(f"  {h[:16]}  {path}")
        return

    if args.plan:
        dev = check_frozen(cfg.study, frozen_inputs(cfg, yaml_path))
        print(f"study {cfg.study}: {len(plan_train_jobs(cfg))} training jobs; frozen-input deviations: {dev or 'none'}")
        for job in plan_train_jobs(cfg):
            h8 = config_hash(resolved_run_config(cfg, job.algo, job.level, job.seed))
            state = classify_run(job, h8, RUNS_DIR)
            print(f"  {job.key:28s} {state.action:7s} {state.run_dir or ''}  {state.reason}")
        return

    run_stages = tuple(s for s in stages if s in ("train", "eval", "baselines"))
    if run_stages:
        manifest = run_study(cfg, parallel=args.parallel, stages=run_stages, allow_deviation=args.allow_deviation,
                             study_yaml=yaml_path)
        failed = [k for k, j in manifest.data["jobs"].items() if j.get("status") not in ("completed",)]
        print(f"jobs: {len(manifest.data['jobs'])}, not completed: {failed or 'none'}")
    if "aggregate" in stages:
        out = write_study_outputs(cfg, results_md=Path(args.results_md) if args.results_md else None)
        print(f"summary: {out['summary']}\nRESULTS: {out['results_md']}\nfigures: {len(out['figures'])}")
    if "replays" in stages:
        written = record_evolution_replays(cfg)
        print(f"evolution replays written: {len(written)}")


if __name__ == "__main__":
    main()

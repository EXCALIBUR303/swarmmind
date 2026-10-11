#!/usr/bin/env python
"""Statistical leak probe (§12.4, §22.4 item 5) on every final actor of a study.

Writes `results/studies/<study>/leak_probe_<level>.json` with one entry per run
(`<algo>_s<seed>`): the paired pre-discovery target-direction bias, its z and
one-sided p, and whether it is flagged. Also prints a POOLED estimate over all
runs (pairs are independent across runs), which has ~3x the per-run power.

Frozen L2 validation maps, probe seed 0, stochastic actors — the parameters the
Phase 7 file was produced with (this script reproduces it byte for byte;
Phase 15 added the script because the original run was not in the repository).

    python scripts/run_leak_probe.py --study main_L2 [--level L2] [--check]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
from marl.analysis.leak_probe import run_leak_probe
from marl.envs.core.config import LevelRegistry
from marl.evaluation.eval_sets import load_eval_set
from marl.experiments.config import load_study_config
from marl.experiments.outputs import planned_runs
from marl.utils.paths import STUDIES_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--level", default="L2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="compare with the stored file instead of writing it")
    args = ap.parse_args()

    cfg = load_study_config(args.study)
    env_cfg = LevelRegistry().get(args.level)
    maps = load_eval_set(args.level, "validation")  # validation only: never held-out maps
    out: dict[str, dict] = {}
    pooled_n = pooled_sum = pooled_var = 0.0
    for r in sorted((r for r in planned_runs(cfg) if r["level"] == args.level and r["run_dir"]),
                    key=lambda r: (r["algo"], r["seed"])):
        actor, _ = load_actor(str(Path(r["run_dir"]) / "actor.pt"))
        res = run_leak_probe(env_cfg, LearnedActorPolicy(actor, stochastic=True), maps, seed=args.seed)
        out[f"{r['algo']}_s{r['seed']}"] = {"n_pairs": res.n_pairs, "diff": res.diff, "z": res.z,
                                            "p_value": res.p_value, "flagged": res.flagged}
        pooled_n += res.n_pairs
        pooled_sum += res.diff * res.n_pairs
        if res.z:  # per-run sample variance recovered from SE = diff / z
            pooled_var += res.n_pairs * (res.diff / res.z) ** 2 * res.n_pairs
    path = STUDIES_DIR / args.study / f"leak_probe_{args.level}.json"
    if args.check:
        same = json.loads(path.read_text()) == out
        print(f"{path}: {'IDENTICAL to a fresh run' if same else 'DIFFERS from a fresh run'}")
        if not same:
            return 1
    else:
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {path}")
    flagged = [k for k, v in out.items() if v["flagged"]]
    mean = pooled_sum / max(pooled_n, 1)
    se = math.sqrt(pooled_var) / max(pooled_n, 1)
    z = mean / se if se else 0.0
    print(f"runs flagged: {flagged or 'none'}; pooled bias over {int(pooled_n)} pairs: {mean:+.4f} "
          f"(SE {se:.4f}, z {z:+.2f}, one-sided p {0.5 * math.erfc(z / math.sqrt(2)):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

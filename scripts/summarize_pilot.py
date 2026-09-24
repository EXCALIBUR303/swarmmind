#!/usr/bin/env python
"""Print a markdown table of a pilot study's validation curves (Phase 6
budget calibration). Validation maps only — the pilot never touches the
held-out sets, and its seeds are disjoint from every study's seeds.

Usage:
    python scripts/summarize_pilot.py --study pilot_L2 --level L2 [--every 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.utils.paths import RUNS_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", default="pilot_L2")
    p.add_argument("--level", default="L2")
    p.add_argument("--every", type=int, default=1, help="print every k-th evaluation point")
    args = p.parse_args()

    curves = {}
    for algo_dir in sorted((RUNS_DIR / args.study).iterdir()):
        for run in sorted((algo_dir / args.level).iterdir()):
            rows = {}  # last record per update wins (same rule as the aggregator)
            for line in (run / "eval_val.jsonl").read_text().splitlines():
                r = json.loads(line)
                rows[int(r["update"])] = r
            ordered = sorted(rows.values(), key=lambda r: r["env_steps"])
            manifest = json.loads((run / "manifest.json").read_text())
            curves[f"{algo_dir.name} (seed {manifest['seed']})"] = ordered

    names = list(curves)
    steps = sorted({r["env_steps"] for rows in curves.values() for r in rows})[:: args.every]
    print("| Env steps | " + " | ".join(f"{n}: success / censored steps" for n in names) + " |")
    print("|---|" + "---|" * len(names))
    for s in steps:
        cells = []
        for n in names:
            r = next((r for r in curves[n] if r["env_steps"] == s), None)
            cells.append("—" if r is None else f"{r['success_rate']:.2f} / {r['censored_mean_steps']:.1f}")
        print(f"| {s:,} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

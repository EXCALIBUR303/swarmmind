#!/usr/bin/env python
"""Aggregate a study into summary.json, figures and docs/RESULTS.md (§34:
RESULTS.md is generated, never hand-edited).

Usage:
    python scripts/aggregate_results.py --study smoke [--results-md docs/RESULTS.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.experiments.config import load_study_config
from marl.experiments.outputs import write_study_outputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--results-md", default=None)
    args = p.parse_args()
    out = write_study_outputs(load_study_config(args.study),
                              results_md=Path(args.results_md) if args.results_md else None)
    print(f"summary: {out['summary']}\nRESULTS: {out['results_md']}")
    for f in out["figures"]:
        print(f"figure: {f}")


if __name__ == "__main__":
    main()

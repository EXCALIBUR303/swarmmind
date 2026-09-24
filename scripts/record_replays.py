#!/usr/bin/env python
"""Record Swarm Evolution replays (0/10/50/100% checkpoints, same maps and
evaluation seed) for a finished study (§26, §27.3).

Usage:
    python scripts/record_replays.py --study smoke --evolution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.experiments.config import load_study_config
from marl.experiments.outputs import record_evolution_replays


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--evolution", action="store_true", required=True,
                   help="record the checkpoint-evolution set (the only mode in v1)")
    args = p.parse_args()
    written = record_evolution_replays(load_study_config(args.study))
    print(f"wrote {len(written)} replays")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()

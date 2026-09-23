#!/usr/bin/env python
"""Generate and freeze the validation/test map sets for every level (§20.3).

Run ONCE per level (or whenever a level's config changes in a way that
requires regenerating its eval sets — which should be rare and always a
deliberate, documented decision, since these files are committed and
treated as immutable by everything downstream).

Usage:
    python scripts/make_eval_maps.py [--levels L1 L2 L3 L4] [--n-val 50] [--n-test 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.envs.core.config import LevelRegistry
from marl.evaluation.eval_sets import generate_test_maps, generate_validation_maps, save_eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4"])
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=100)
    args = parser.parse_args()

    registry = LevelRegistry()
    for level in args.levels:
        cfg = registry.get(level)
        print(f"[{level}] generating {args.n_val} validation maps...")
        val_maps = generate_validation_maps(cfg, n=args.n_val)
        val_path = save_eval_set(val_maps, level, "validation")
        print(f"[{level}]   -> {val_path}")

        print(f"[{level}] generating {args.n_test} test maps...")
        test_maps = generate_test_maps(cfg, n=args.n_test)
        test_path = save_eval_set(test_maps, level, "test")
        print(f"[{level}]   -> {test_path}")

        # Sanity: no hash overlap between validation and test for this level.
        val_hashes = {m.content_hash for m in val_maps}
        test_hashes = {m.content_hash for m in test_maps}
        overlap = val_hashes & test_hashes
        if overlap:
            print(f"[{level}] WARNING: {len(overlap)} maps appear in BOTH validation and test!")
        else:
            print(f"[{level}] OK: validation and test sets are disjoint ({len(val_hashes)}/{len(test_hashes)} unique).")

    print("Done.")


if __name__ == "__main__":
    main()

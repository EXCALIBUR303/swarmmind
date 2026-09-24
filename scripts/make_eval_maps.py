#!/usr/bin/env python
"""Generate and freeze the validation/test map sets for every level (§20.3).

Run ONCE per level (or whenever a level's config changes in a way that
requires regenerating its eval sets — which should be rare and always a
deliberate, documented decision, since these files are committed and
treated as immutable by everything downstream).

Usage:
    python scripts/make_eval_maps.py [--levels L1 L2 L3 L4] [--n-val 50] [--n-test 100]
    python scripts/make_eval_maps.py --structured --levels L2   # freeze handmade G2 maps (§20.4)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.envs.core.config import LevelRegistry
from marl.evaluation.eval_sets import (
    build_structured_set,
    eval_set_exists,
    generate_test_maps,
    generate_validation_maps,
    load_eval_set,
    save_eval_set,
    write_maps_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4"])
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing frozen sets (invalidates calibration and every result evaluated on them)",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="freeze the handmade structured split from configs/maps/structured/<level>/*.txt instead",
    )
    args = parser.parse_args()

    registry = LevelRegistry()
    if args.structured:
        _freeze_structured(registry, args.levels, args.force)
        return
    existing = [lvl for lvl in args.levels for split in ("validation", "test") if eval_set_exists(lvl, split)]
    if existing and not args.force:
        sys.exit(
            f"refusing to overwrite frozen eval sets for {sorted(set(existing))}: they are immutable "
            "(§20.3), and calibration + all results were evaluated on them. Pass --force only for a "
            "deliberate, documented version bump."
        )

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

    print(f"manifest -> {write_maps_manifest()}")
    print("Done.")


def _freeze_structured(registry: LevelRegistry, levels: list[str], force: bool) -> None:
    existing = [lvl for lvl in levels if eval_set_exists(lvl, "structured")]
    if existing and not force:
        sys.exit(f"refusing to overwrite frozen structured sets for {existing} (§20.3); pass --force deliberately.")
    procedural = {
        m.content_hash
        for lvl in registry.available_levels()
        for split in ("validation", "test")
        if eval_set_exists(lvl, split)
        for m in load_eval_set(lvl, split)
    }
    for level in levels:
        maps = build_structured_set(registry.get(level), level)
        clash = {m.content_hash for m in maps} & procedural
        if clash:
            sys.exit(f"[{level}] structured maps collide with procedural eval maps: {clash}")
        path = save_eval_set(maps, level, "structured")
        print(f"[{level}] froze {len(maps)} structured maps -> {path}")
    print(f"manifest -> {write_maps_manifest()}")


if __name__ == "__main__":
    main()

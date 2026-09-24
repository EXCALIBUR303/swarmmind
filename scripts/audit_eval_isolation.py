#!/usr/bin/env python
"""Retroactive §20.3 audit: did a training run ever see a frozen eval map?

A run's training map stream is fully determined by its run seed (the first
`n_envs` generators spawned from `SeedStreams(run_seed)`, one seed drawn
per episode reset). This replays that stream up to a proven upper bound on
the number of resets and checks every map's content hash against the
frozen validation + test manifests of every level. Needed for runs made
before the in-trainer isolation guard existed (Phase 4 IPPO, Phase 5
MAPPO gate runs); runs made after it record
`training_maps_rejected_as_eval` in their manifest instead.

Upper bound: an env slot resets at most once per `min_episode_len` env
steps, so at most ceil(steps_per_env / min_episode_len) + 1 draws.

Usage:
    python scripts/audit_eval_isolation.py --level L1 --seeds 0 1 2 \
        --n-envs 16 --total-env-steps 300000 --min-episode-len 3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.envs.core.config import LevelRegistry
from marl.envs.core.mapgen import generate_map
from marl.evaluation.eval_sets import frozen_hash_manifest
from marl.utils.seeding import SeedStreams

MAX_SEED = 2**31 - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--total-env-steps", type=int, required=True)
    parser.add_argument("--min-episode-len", type=int, required=True)
    args = parser.parse_args()

    registry = LevelRegistry()
    cfg = registry.get(args.level)
    banned = frozen_hash_manifest(registry.available_levels())
    steps_per_env = math.ceil(args.total_env_steps / args.n_envs)
    draws_per_env = math.ceil(steps_per_env / args.min_episode_len) + 1

    total_hits = 0
    for run_seed in args.seeds:
        gens = SeedStreams(run_seed).spawn_generators(args.n_envs)
        hits = 0
        for gen in gens:
            for _ in range(draws_per_env):
                spec = generate_map(cfg, int(gen.integers(0, MAX_SEED)))
                if spec.content_hash in banned:
                    hits += 1
        total_hits += hits
        print(
            f"[{args.level}] run_seed={run_seed}: checked {args.n_envs * draws_per_env} candidate "
            f"training maps (upper bound) against {len(banned)} frozen eval hashes -> {hits} collisions"
        )
    print("CLEAN" if total_hits == 0 else f"CONTAMINATED: {total_hits} collisions")
    sys.exit(0 if total_hits == 0 else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run a baseline policy over a frozen map set and report aggregate metrics.

Phase 3 scope: Random and heuristic only (learned checkpoints arrive in
Phase 4+, at which point this script gains a `--checkpoint` option without
changing its aggregation logic).

Usage:
    python scripts/evaluate.py --level L2 --policy heuristic --split validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy
from marl.algorithms.baselines.random_policy import RandomPolicy
from marl.analysis.metrics import compute_episode_metrics
from marl.envs.core.config import LevelRegistry
from marl.evaluation.eval_sets import load_eval_set
from marl.evaluation.evaluator import run_episode
from marl.utils.hashing import sha256_hex

POLICIES = {
    "random": lambda: RandomPolicy(),
    "heuristic": lambda: FrontierHeuristicPolicy(),
}


def evaluate(level: str, policy_name: str, split: str, episodes_per_map: int = 1) -> dict:
    registry = LevelRegistry()
    cfg = registry.get(level)
    maps = load_eval_set(level, split)
    policy = POLICIES[policy_name]()

    successes = 0
    steps_list = []
    censored_steps_list = []
    coverage_list = []
    n_episodes = 0

    for map_spec in maps:
        for ep in range(episodes_per_map):
            # A deterministic seed derived from (map_id, episode index) via
            # sha256 — NOT Python's built-in hash(), which is randomized
            # per-process (PYTHONHASHSEED) and would make every run of this
            # script produce different episode outcomes (§34 reproducibility).
            digest = sha256_hex(f"{map_spec.id}:{ep}")
            episode_seed = int(digest[:8], 16)
            rng = np.random.default_rng(episode_seed)
            result = run_episode(cfg, policy, rng, map_spec=map_spec)
            metrics = compute_episode_metrics(result.trace, result.n_reachable_free)
            n_episodes += 1
            if metrics.success:
                successes += 1
                steps_list.append(metrics.steps_to_completion)
                censored_steps_list.append(metrics.steps_to_completion)
            else:
                censored_steps_list.append(cfg.max_steps)
            coverage_list.append(metrics.coverage_final)

    return {
        "level": level,
        "policy": policy_name,
        "split": split,
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "mean_steps_on_success": float(np.mean(steps_list)) if steps_list else None,
        "p90_steps_on_success": float(np.percentile(steps_list, 90)) if steps_list else None,
        "censored_mean_steps": float(np.mean(censored_steps_list)),
        "mean_coverage": float(np.mean(coverage_list)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True)
    parser.add_argument("--policy", choices=list(POLICIES), required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test", "structured"])
    parser.add_argument("--episodes-per-map", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of a summary line")
    args = parser.parse_args()

    result = evaluate(args.level, args.policy, args.split, args.episodes_per_map)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"[{result['level']}] {result['policy']} on {result['split']} "
            f"({result['n_episodes']} episodes): success={result['success_rate']:.3f}, "
            f"mean_steps={result['mean_steps_on_success']}, "
            f"p90_steps={result['p90_steps_on_success']}, "
            f"coverage={result['mean_coverage']:.3f}"
        )


if __name__ == "__main__":
    main()

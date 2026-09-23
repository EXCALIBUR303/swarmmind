#!/usr/bin/env python
"""Single-process environment throughput benchmark (§35).

Usage:
    python scripts/bench_env.py [--level L2] [--steps 20000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from marl.envs.core.config import LevelRegistry
from marl.envs.core.world import SwarmWorld


def bench(level: str, n_steps: int) -> None:
    registry = LevelRegistry()
    cfg = registry.get(level)
    world = SwarmWorld(cfg)
    world.reset(seed=0)
    rng = np.random.default_rng(0)

    # Full loop: observation + mask construction + step (what a rollout
    # collection loop actually does every tick).
    start = time.perf_counter()
    for _ in range(n_steps):
        _, _, masks = world.all_actor_obs()
        actions = np.array(
            [rng.choice(np.where(masks[i] == 1)[0]) if np.any(masks[i] == 1) else 0 for i in range(cfg.n_robots)]
        )
        result = world.step(actions)
        if result.terminated or result.truncated:
            world.reset(seed=int(rng.integers(0, 1_000_000)))
    elapsed = time.perf_counter() - start
    print(
        f"[{level}] {n_steps} env steps WITH obs+mask construction: "
        f"{elapsed:.2f}s -> {n_steps / elapsed:.0f} env-steps/s"
    )
    if n_steps / elapsed < 1000:
        print(f"[{level}] WARNING: throughput below the §35 flag threshold (1000 env-steps/s)")

    # step()-only: dynamics + reward + sense, no observation construction.
    world.reset(seed=1)
    start = time.perf_counter()
    for _ in range(n_steps):
        actions = rng.integers(0, 5, size=cfg.n_robots)
        result = world.step(actions)
        if result.terminated or result.truncated:
            world.reset(seed=int(rng.integers(0, 1_000_000)))
    elapsed2 = time.perf_counter() - start
    print(
        f"[{level}] {n_steps} step()-only calls: "
        f"{elapsed2:.2f}s -> {n_steps / elapsed2:.0f} env-steps/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="L2")
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args()
    bench(args.level, args.steps)


if __name__ == "__main__":
    main()

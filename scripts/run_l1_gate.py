#!/usr/bin/env python
"""Phase 4/5 L1 learning-gate run for one (algo, seed), resumable.

Uses the exact config of the Phase 4 IPPO gate runs (n_envs=16,
rollout_len=64, validation every 10 updates on 50 frozen L1 validation
maps) so IPPO and MAPPO gate results are directly comparable. Re-running
the same command after an interruption resumes from `latest.pt`.

Usage:
    python scripts/run_l1_gate.py --algo mappo --seed 0 [--total-env-steps 300000] [--threads 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from marl.algorithms.common.config import AlgoConfig
from marl.envs.core.config import LevelRegistry
from marl.training.trainer import train
from marl.utils.config_loader import load_yaml
from marl.utils.paths import CONFIGS_DIR, RUNS_DIR


def _agent_builder(algo: str):
    if algo == "ippo":
        from marl.algorithms.ippo.agent import IPPOAgent

        return IPPOAgent
    from marl.algorithms.mappo.agent import MAPPOAgent

    return MAPPOAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=["ippo", "mappo"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-env-steps", type=int, default=300_000)
    parser.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = torch default)")
    parser.add_argument("--study", default="l1_gate")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    env_cfg = LevelRegistry().get("L1")
    algo_cfg = AlgoConfig.model_validate(load_yaml(CONFIGS_DIR / "algo" / f"{args.algo}.yaml"))
    algo_cfg.ppo.n_envs = 16
    algo_cfg.ppo.rollout_len = 64
    algo_cfg.eval.interval_updates = 10
    algo_cfg.eval.n_val_maps = 50
    algo_cfg.checkpoint.fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

    run_dir = RUNS_DIR / args.study / args.algo / "L1" / f"seed{args.seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    latest = run_dir / "checkpoints" / "latest.pt"
    resume_from = latest if latest.exists() else None

    start = time.perf_counter()
    result = train(
        env_cfg,
        algo_cfg,
        _agent_builder(args.algo),
        run_seed=args.seed,
        total_env_steps=args.total_env_steps,
        run_dir=run_dir,
        study=args.study,
        algo_name=args.algo,
        level="L1",
        resume_from=resume_from,
    )
    elapsed = time.perf_counter() - start
    print(f"ELAPSED {elapsed:.1f}s env_steps {result.global_env_steps} status {result.status}")
    print(f"FINAL_VAL {result.final_val_metrics}")


if __name__ == "__main__":
    main()

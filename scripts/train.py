#!/usr/bin/env python
"""Train IPPO (or MAPPO, from Phase 5) on a level (§21.1).

Usage:
    python scripts/train.py --level L1 --algo ippo --seed 0 --total-env-steps 200000 \
        [--study smoke] [--set ppo.lr_actor=1e-4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.algorithms.common.config import AlgoConfig
from marl.envs.core.config import LevelRegistry
from marl.training.trainer import train
from marl.utils.config_loader import apply_overrides, load_yaml
from marl.utils.paths import CONFIGS_DIR


def _get_agent_builder(algo: str):
    if algo == "ippo":
        from marl.algorithms.ippo.agent import IPPOAgent

        return IPPOAgent
    if algo == "mappo":
        from marl.algorithms.mappo.agent import MAPPOAgent  # Phase 5

        return MAPPOAgent
    raise ValueError(f"unknown algo: {algo}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True)
    parser.add_argument("--algo", required=True, choices=["ippo", "mappo"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-env-steps", type=int, required=True)
    parser.add_argument("--study", default="adhoc")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--runs-dir", default=None, help="parent directory for new run dirs (default: results/runs)")
    parser.add_argument("--resume", default=None, help="path to a checkpoint (e.g. .../checkpoints/latest.pt) to resume from")
    parser.add_argument("--run-dir", default=None, help="required with --resume: the existing run directory to continue writing into")
    args = parser.parse_args()

    from marl.experiments.workers import _raise_on_sigterm

    _raise_on_sigterm()

    env_cfg = LevelRegistry().get(args.level)
    algo_raw = load_yaml(CONFIGS_DIR / "algo" / f"{args.algo}.yaml")
    if args.overrides:
        algo_raw = apply_overrides(algo_raw, args.overrides)
    algo_cfg = AlgoConfig.model_validate(algo_raw)

    from pathlib import Path as _Path

    from marl.training.run_dir import make_run_dir

    if args.resume:
        if not args.run_dir:
            parser.error("--resume requires --run-dir (the existing run directory to continue)")
        run_dir = _Path(args.run_dir)
    else:
        resolved = {"env": env_cfg.model_dump(), "algo": algo_cfg.model_dump(), "run_seed": args.seed}
        run_dir = make_run_dir(
            args.study, args.algo, args.level, args.seed, resolved,
            base_dir=_Path(args.runs_dir) if args.runs_dir else None,
        )
    print(f"run_dir: {run_dir}")

    result = train(
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
        agent_builder=_get_agent_builder(args.algo),
        run_seed=args.seed,
        total_env_steps=args.total_env_steps,
        run_dir=run_dir,
        study=args.study,
        algo_name=args.algo,
        level=args.level,
        resume_from=_Path(args.resume) if args.resume else None,
    )

    print(f"status: {result.status}")
    print(f"env_steps: {result.global_env_steps}, agent_steps: {result.global_agent_steps}, updates: {result.updates}")
    print(f"final_val_metrics: {result.final_val_metrics}")


if __name__ == "__main__":
    main()

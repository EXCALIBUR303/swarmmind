#!/usr/bin/env python
"""Re-export an actor-only checkpoint from a training checkpoint (§15.10).

Useful when you want an inference-only artifact from a specific
historical checkpoint (not just the final one, which the trainer already
exports automatically at `run_dir/actor.pt`).

Usage:
    python scripts/export_actor.py --checkpoint results/runs/.../checkpoints/step_100000.pt \
        --out models/ippo_L1_s0/actor.pt --algo ippo --level L1 --hidden 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.networks.actor import ActorPolicy
from marl.training.checkpoint import export_actor, load_checkpoint


class _ActorOnlyShim:
    """Minimal object satisfying `export_actor`'s `agent.actor` access
    without constructing a full PPOAgent (critic/optimizers not needed)."""

    def __init__(self, actor: ActorPolicy) -> None:
        self.actor = actor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    actor = ActorPolicy(grid_size=args.grid_size, hidden=args.hidden)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    export_actor(
        args.out,
        _ActorOnlyShim(actor),
        meta={
            "grid_size": args.grid_size,
            "hidden": args.hidden,
            "algo": args.algo,
            "level": args.level,
            "source_checkpoint": args.checkpoint,
            "global_env_steps": ckpt.get("global_env_steps"),
        },
    )
    print(f"exported actor -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""PRELIMINARY L1 learning curves, IPPO vs MAPPO (Phase 5 deliverable).

Not the study (that's Phase 7, on L2, with pre-registered statistics):
L1 saturates for both algorithms, n=3 seeds, no claims are made from it.

Usage: python scripts/plot_l1_gate.py [--study l1_gate_v2] [--out docs/figures/phase5_l1_preliminary.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marl.envs.core.config import LevelRegistry
from marl.networks.actor import ActorPolicy
from marl.training.checkpoint import load_checkpoint
from marl.training.validate import evaluate_on_validation
from marl.utils.paths import REPO_ROOT, RUNS_DIR

# Reference categorical slots 1-2 (validated: CVD ΔE 24.7, normal ΔE 33.6).
COLORS = {"ippo": "#2a78d6", "mappo": "#eb6834"}
LABELS = {"ippo": "IPPO", "mappo": "MAPPO"}
INK, INK_MUTED, GRID, SURFACE = "#1b1b1a", "#6b6a63", "#e6e5df", "#fcfcfb"
# Phase 3 calibration (docs/EXPERIMENTS.md), same 50 frozen L1 validation maps.
RANDOM_SUCCESS, RANDOM_STEPS = 0.58, 71.48


def load_curves(study: str, algo: str, env_cfg) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves = []
    for seed in (0, 1, 2):
        run = RUNS_DIR / study / algo / "L1" / f"seed{seed}"
        with open(run / "eval_val.jsonl") as f:
            rows = [json.loads(line) for line in f]
        actor0 = ActorPolicy()
        actor0.load_state_dict(load_checkpoint(run / "checkpoints" / "step_0.pt")["actor_state_dict"])
        m0 = evaluate_on_validation(env_cfg, actor0, "L1", n_maps=50)
        x = np.array([0] + [r["env_steps"] for r in rows]) / 1000
        succ = np.array([m0["success_rate"]] + [r["success_rate"] for r in rows])
        steps = np.array([m0["censored_mean_steps"]] + [r["censored_mean_steps"] for r in rows])
        curves.append((x, succ, steps))
    return curves


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", default="l1_gate_v2")
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "figures" / "phase5_l1_preliminary.png"))
    args = p.parse_args()

    env_cfg = LevelRegistry().get("L1")
    data = {a: load_curves(args.study, a, env_cfg) for a in ("ippo", "mappo")}

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [
        (axes[0], 1, "Validation success rate", (0, 1.02), RANDOM_SUCCESS, "Random"),
        (axes[1], 2, "Censored mean steps to completion (lower is better)", (0, 100), RANDOM_STEPS, "Random"),
    ]
    for ax, idx, title, ylim, ref, ref_label in panels:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axhline(ref, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(302, ref, f" {ref_label}", va="center", ha="left", color=INK_MUTED, fontsize=9)
        means = {}
        for algo, curves in data.items():
            for x, *ys in curves:
                ax.plot(x, ys[idx - 1], color=COLORS[algo], linewidth=0.8, alpha=0.35)
            x = curves[0][0]
            means[algo] = mean = np.mean([c[idx] for c in curves], axis=0)
            ax.plot(x, mean, color=COLORS[algo], linewidth=2.0, label=f"{LABELS[algo]} (mean of 3 seeds)")
        # Direct labels where the curves are furthest apart (they converge at
        # the right edge, so end-of-line labels would overprint each other).
        k = int(np.argmax(np.abs(means["ippo"] - means["mappo"])))
        hi, lo = sorted(means, key=lambda a: means[a][k], reverse=True)
        span = ylim[1] - ylim[0]
        # Empty space above a RISING curve is to its left; above a FALLING one, to its right.
        rising = means[hi][-1] > means[hi][0]
        hi_x = max(x[k] - 16, 2) if rising else x[k] + 8
        ax.annotate(LABELS[hi], (x[k], means[hi][k]), xytext=(hi_x, means[hi][k] + 0.08 * span),
                    color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 0.6})
        ax.annotate(LABELS[lo], (x[k], means[lo][k]), xytext=(x[k] + 8, means[lo][k] - 0.09 * span),
                    color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 0.6})
        ax.set_xlim(0, 300)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Environment steps (thousands)")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "PRELIMINARY — L1 (8×8, 2 robots), 3 seeds each, 50 frozen validation maps. "
        "Thin lines: individual seeds. Not the study; no comparative claims.",
        x=0.01, ha="left", fontsize=9.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0, 0.97, 0.94))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

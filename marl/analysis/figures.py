"""Static study figures for the written report (§22.2, §26), drawn only
from `summary.json` so the figures and RESULTS.md can never disagree.

Palette: categorical slots validated in Phase 5 (IPPO #2a78d6, MAPPO
#eb6834; CVD ΔE 24.7, normal ΔE 33.6). Baselines are drawn in neutral ink
(dashed/dotted), never in a categorical hue, so they read as references.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"ippo": "#2a78d6", "mappo": "#eb6834"}
LABELS = {"ippo": "IPPO", "mappo": "MAPPO", "random": "Random", "heuristic": "Heuristic"}
INK, INK_MUTED, GRID, SURFACE = "#1b1b1a", "#6b6a63", "#e6e5df", "#fcfcfb"
METRIC_LABELS = {
    "success_rate": "Success rate",
    "censored_steps": "Censored mean steps (lower is better)",
    "coverage": "Final coverage",
    "collisions_per_100_agent_steps": "Collisions per 100 agent-steps",
}


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)


def _banner(fig: Any, summary: dict[str, Any], text: str) -> None:
    import textwrap

    prefix = "SMOKE — infrastructure dry run, not evidence. " if summary.get("smoke") else ""
    width = int(fig.get_figwidth() * 13)  # ~characters that fit at 9.5 pt
    fig.suptitle(textwrap.fill(prefix + text, width), x=0.01, ha="left", fontsize=9.5, color=INK_MUTED)


def _baseline_mean(methods: dict[str, Any], kind: str, split: str, metric: str) -> float | None:
    blk = methods.get(kind, {}).get("splits", {}).get(split, {}).get("stochastic")
    iv = blk.get(metric, {}).get("interval") if blk else None
    return iv["mean"] if iv else None


def plot_learning_curves(summary: dict[str, Any], level: str, path: Path) -> Path | None:
    methods = summary["levels"][level]["methods"]
    curves = {a: methods.get(a, {}).get("validation_curve") for a in ("ippo", "mappo")}
    if not any(curves.values()):
        return None
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax)
    x_max = 0.0
    for algo, c in curves.items():
        if not c:
            continue
        x = np.asarray(c["grid_env_steps"], dtype=float) / 1e3
        x_max = max(x_max, float(x[-1]))
        for row in c["per_seed"]:
            y = np.array([np.nan if v is None else v for v in row], dtype=float)
            ax.plot(x, y, color=COLORS[algo], linewidth=0.8, alpha=0.35)
        mean = np.array([np.nan if v is None else v for v in c["mean"]], dtype=float)
        lo = np.array([np.nan if v is None else v for v in c["lo"]], dtype=float)
        hi = np.array([np.nan if v is None else v for v in c["hi"]], dtype=float)
        ax.fill_between(x, lo, hi, color=COLORS[algo], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=COLORS[algo], linewidth=2.0,
                label=f"{LABELS[algo]} (mean of {len(c['per_seed'])} seeds, 95% CI)")
    for kind, style in (("random", (0, (4, 3))), ("heuristic", (0, (1, 2)))):
        ref = _baseline_mean(methods, kind, "validation", "success_rate")
        if ref is not None:
            ax.axhline(ref, color=INK_MUTED, linestyle=style, linewidth=1.2)
            ax.text(x_max * 1.005, ref, f" {LABELS[kind]}", va="center", ha="left", color=INK_MUTED, fontsize=9)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Environment steps (thousands)", color=INK)
    ax.set_title(f"{level} validation success during training (thin lines: individual seeds)",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _banner(fig, summary, f"Study {summary['study']}: validation maps only; held-out results are in the dot plots.")
    fig.tight_layout(rect=(0, 0, 0.95, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_heldout_dots(summary: dict[str, Any], level: str, split: str, metric: str, path: Path) -> Path | None:
    """Per-seed held-out values (the unit of analysis), mean ± 95% t-CI."""
    methods = summary["levels"][level]["methods"]
    order = [m for m in ("random", "ippo", "mappo") if methods.get(m, {}).get("splits", {}).get(split)]
    if not order:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor=SURFACE)
    _style(ax)
    rng = np.random.default_rng(0)  # deterministic jitter
    for i, m in enumerate(order):
        blk = methods[m]["splits"][split]["stochastic"]
        vals = [v for v in blk[metric]["per_seed"] if v is not None]
        color = COLORS.get(m, INK_MUTED)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=48, color=color, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=None)
        iv = blk[metric]["interval"]
        if iv and iv["lo"] is not None:
            ax.errorbar(i + 0.3, iv["mean"], yerr=[[iv["mean"] - iv["lo"]], [iv["hi"] - iv["mean"]]],
                        fmt="o", color=INK, markersize=5, capsize=4, linewidth=1.2, zorder=4)
        elif iv:
            ax.plot(i + 0.3, iv["mean"], "o", color=INK, markersize=5, zorder=4)
    ref = _baseline_mean(methods, "heuristic", split, metric)
    if ref is not None:
        ax.axhline(ref, color=INK_MUTED, linestyle=(0, (1, 2)), linewidth=1.2)
        # Left edge: the learned methods (right) sit near the heuristic's level.
        ax.text(-0.45, ref, "Heuristic (reference)", va="bottom", ha="left", color=INK_MUTED, fontsize=8.5)
    ax.set_xticks(range(len(order)), [LABELS[m] for m in order], color=INK)
    ax.set_xlim(-0.5, len(order) - 0.3)
    if metric == "success_rate":
        ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{level} {split} maps: {METRIC_LABELS.get(metric, metric)}", loc="left", fontsize=10.5, color=INK)
    _banner(fig, summary, "Dots: one per seed (unit of analysis). Black: mean ± 95% t-CI.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_heatmap(counts: list[list[int]], obstacles: list[list[bool]], title: str, path: Path) -> Path:
    """A sequential single-hue heatmap (§26: exploration/collision heatmaps),
    with obstacle cells hatched out so the map layout stays legible."""
    arr = np.array(counts, dtype=float)
    obs = np.array(obstacles, dtype=bool)
    fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    masked = np.ma.masked_where(obs, arr)
    im = ax.imshow(masked, cmap="Blues", vmin=0)
    ax.imshow(np.ma.masked_where(~obs, np.ones_like(arr)), cmap="Greys", vmin=0, vmax=1, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_der_vs_coverage(coverage: list[float], der: list[float], algo: str, title: str, path: Path) -> Path:
    """§24.1's "DER vs. coverage curve" — one illustrative episode, not an
    aggregate statistic (aggregates are DER@50, in the dot-style tables)."""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    _style(ax)
    ax.plot(coverage, der, color=COLORS.get(algo, INK), linewidth=2.0)
    ax.set_xlabel("Coverage C(t)", color=INK)
    ax.set_ylabel("DER", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.05, max(der) * 1.1) if der else 1.0)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_discovery_timeline(timeline: list[dict[str, Any]], max_t: int, title: str, path: Path) -> Path:
    """A Gantt-style strip: discovery -> rescue latency per target (§26)."""
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.4 * max(len(timeline), 1)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, t in enumerate(timeline):
        d, r = t["t_discovered"], t["t_rescued"]
        if d is not None and r is not None:
            ax.barh(i, r - d, left=d, color=COLORS.get("mappo", INK), alpha=0.7)
            ax.plot(r, i, "D", color=INK, markersize=5)
        if d is not None:
            ax.plot(d, i, "o", color=INK_MUTED, markersize=6)
    ax.set_yticks(range(len(timeline)), [f"target {t['target']}" for t in timeline], color=INK)
    ax.set_xlim(0, max_t)
    ax.set_xlabel("Step", color=INK)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def make_study_figures(summary: dict[str, Any], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for level, data in summary["levels"].items():
        p = plot_learning_curves(summary, level, out_dir / f"{level}_validation_curves.png")
        if p:
            paths.append(p)
        splits = {s for m in data["methods"].values() for s in m.get("splits", {}) if s != "validation"}
        for split in sorted(splits):
            for metric in ("success_rate", "censored_steps"):
                p = plot_heldout_dots(summary, level, split, metric, out_dir / f"{level}_{split}_{metric}.png")
                if p:
                    paths.append(p)
    return paths

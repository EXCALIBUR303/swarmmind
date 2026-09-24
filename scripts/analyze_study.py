#!/usr/bin/env python
"""Phase 8 coordination and behaviour analysis (§24, §26) for a finished
study: writes `results/studies/<study>/analysis/<level>_coordination.json`
(consumed automatically by `scripts/aggregate_results.py` to fill in H3 and
the §24.4 criteria — re-run that after this to refresh RESULTS.md), plus
showcase heatmaps/timelines/DER-vs-coverage figures.

Sample size: coordination analysis (team + isolation-null runs) is
run on a SUBSET of the frozen test set (`--n-maps`, default 25 of the
100), not the full 100-map x 2-seed set the main performance results use.
The isolation null alone multiplies the episode count by n_robots, so the
full set would cost ~4x more compute for a secondary, descriptive
analysis; 25 maps x 2 eval seeds = 50 episodes/seed is a reasoned,
documented reduction (this is stated in the Phase 8 report), not silent.

Usage:
    python scripts/analyze_study.py --study main_L2 [--n-maps 25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from marl.algorithms.baselines.random_policy import RandomPolicy
from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
from marl.analysis.behaviour import accumulate_heatmaps, episode_behaviour, obstacle_mask
from marl.analysis.figures import plot_der_vs_coverage, plot_discovery_timeline, plot_heatmap
from marl.envs.core.config import LevelRegistry
from marl.evaluation.ablation import run_ablation_for_checkpoint, summarize_ablation
from marl.evaluation.coordination import run_coordination_for_seed
from marl.evaluation.eval_sets import eval_set_exists, load_eval_set
from marl.evaluation.evaluator import run_episode
from marl.evaluation.final_eval import episode_seed
from marl.experiments.config import load_study_config
from marl.experiments.outputs import planned_runs
from marl.utils.paths import DOCS_DIR, STUDIES_DIR


def analyze_level(
    cfg,
    level: str,
    n_maps: int,
    n_showcase_maps: int,
    n_showcase_episodes: int,
    with_isolation: bool,
    log,
) -> dict:
    env_cfg = LevelRegistry().get(level)
    eval_seeds = cfg.final_eval.eval_seeds
    test_maps = load_eval_set(level, "test")[:n_maps]
    runs = [r for r in planned_runs(cfg) if r["level"] == level]

    methods: dict[str, list[dict]] = {}
    ablation: dict[str, dict] = {}
    for algo in cfg.algos:
        entries, ablation_entries = [], []
        for r in sorted((r for r in runs if r["algo"] == algo), key=lambda r: r["seed"]):
            if r["run_dir"] is None:
                log(f"  [skip] {algo}/L{level}/s{r['seed']}: run not completed")
                continue
            actor, _meta = load_actor(str(Path(r["run_dir"]) / "actor.pt"))

            def make_policy(a=actor):
                return LearnedActorPolicy(a, stochastic=True)

            coord = run_coordination_for_seed(env_cfg, make_policy, test_maps, eval_seeds, with_isolation)
            entries.append({"seed": r["seed"], **coord})
            log(f"  [{algo} s{r['seed']}] der={coord['der']:.3f} "
                f"der50={coord['der_at_50']['mean']} der_iso={coord['der_iso']}")

            ablation_entries.append(run_ablation_for_checkpoint(env_cfg, actor, test_maps, eval_seeds))
        methods[algo] = entries
        if ablation_entries:
            ablation[algo] = summarize_ablation(ablation_entries)

    random_entries = []
    for s in cfg.baselines.random.eval_seeds:
        coord = run_coordination_for_seed(
            env_cfg, RandomPolicy, test_maps, eval_seeds, with_isolation, sampling_seed=s
        )
        random_entries.append({"sampling_seed": s, **coord})
        log(f"  [random s{s}] der={coord['der']:.3f} der50={coord['der_at_50']['mean']} der_iso={coord['der_iso']}")
    methods["random"] = random_entries

    return {
        "level": level,
        "protocol": {"n_maps": n_maps, "eval_seeds": list(eval_seeds), "with_isolation": with_isolation},
        "methods": methods,
        "ablation": ablation,
    }


def write_showcase(cfg, level: str, n_showcase_maps: int, n_showcase_episodes: int, out_dir: Path, log) -> list[str]:
    """Heatmaps, one DER-vs-coverage curve, and one discovery timeline per
    algorithm, on a handful of structured (interpretable) showcase maps
    using the FIRST completed seed's final checkpoint — illustrative
    behaviour data (§26), not a statistic with a CI."""
    env_cfg = LevelRegistry().get(level)
    if not eval_set_exists(level, "structured"):
        log(f"  no structured set for {level}; skipping showcase figures")
        return []
    showcase_maps = load_eval_set(level, "structured")[:n_showcase_maps]
    runs = [r for r in planned_runs(cfg) if r["level"] == level and r["run_dir"] is not None]

    written: list[str] = []
    for algo in cfg.algos:
        algo_runs = sorted((r for r in runs if r["algo"] == algo), key=lambda r: r["seed"])
        if not algo_runs:
            continue
        run_dir = algo_runs[0]["run_dir"]
        actor, _meta = load_actor(str(Path(run_dir) / "actor.pt"))
        policy = LearnedActorPolicy(actor, stochastic=True)

        for m in showcase_maps:
            traces = []
            for ep in range(n_showcase_episodes):
                rng = np.random.default_rng(episode_seed(f"{m.id}:showcase", ep))
                result = run_episode(env_cfg, policy, rng, map_spec=m)
                traces.append(result.trace)
            heat = accumulate_heatmaps(traces, env_cfg.height, env_cfg.width)
            obstacles = obstacle_mask(m.terrain_array())
            data = {"map_id": m.id, "n_episodes": n_showcase_episodes, "obstacles": obstacles, **heat}
            out_json = out_dir / f"{level}_{algo}_{m.id}_heatmap.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(data))
            written.append(str(out_json))

            plot_heatmap(heat["visit_counts"], obstacles, f"{algo.upper()} visit density — {m.id}",
                        DOCS_DIR / "figures" / cfg.study / f"{level}_{algo}_{m.id}_visits.png")
            plot_heatmap(heat["collision_counts"], obstacles, f"{algo.upper()} collisions — {m.id}",
                        DOCS_DIR / "figures" / cfg.study / f"{level}_{algo}_{m.id}_collisions.png")

        # One example episode (first showcase map, eval seed 0) for the
        # DER-vs-coverage curve and the discovery timeline.
        example_map = showcase_maps[0]
        rng = np.random.default_rng(episode_seed(example_map.id, 0))
        result = run_episode(env_cfg, policy, rng, map_spec=example_map)
        bundle = episode_behaviour(result.trace, result.n_reachable_free, env_cfg.sensor_radius)
        out_json = out_dir / f"{level}_{algo}_example_episode.json"
        out_json.write_text(json.dumps(bundle))
        written.append(str(out_json))
        plot_der_vs_coverage(bundle["coverage_over_time"], bundle["der_over_time"], algo,
                             f"{algo.upper()} DER vs. coverage — {example_map.id}",
                             DOCS_DIR / "figures" / cfg.study / f"{level}_{algo}_der_vs_coverage.png")
        plot_discovery_timeline(bundle["discovery_timeline"], env_cfg.max_steps,
                                f"{algo.upper()} discovery/rescue timeline — {example_map.id}",
                                DOCS_DIR / "figures" / cfg.study / f"{level}_{algo}_discovery_timeline.png")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", required=True)
    p.add_argument("--n-maps", type=int, default=25, help="test maps used for coordination/isolation-null analysis")
    p.add_argument("--n-showcase-maps", type=int, default=3)
    p.add_argument("--n-showcase-episodes", type=int, default=20)
    p.add_argument("--no-isolation", action="store_true", help="skip the isolation-null run (faster dry run)")
    args = p.parse_args()

    cfg = load_study_config(args.study)
    out_dir = STUDIES_DIR / cfg.study / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for level in cfg.levels:
        print(f"[{level}] coordination analysis ({args.n_maps} maps x {cfg.final_eval.eval_seeds} eval seeds)...")
        coord = analyze_level(
            cfg, level, args.n_maps, args.n_showcase_maps, args.n_showcase_episodes,
            with_isolation=not args.no_isolation, log=print,
        )
        (out_dir / f"{level}_coordination.json").write_text(json.dumps(coord, indent=1))
        print(f"  -> {out_dir / f'{level}_coordination.json'}")

        print(f"[{level}] showcase behaviour figures...")
        written = write_showcase(cfg, level, args.n_showcase_maps, args.n_showcase_episodes, out_dir, log=print)
        print(f"  wrote {len(written)} showcase files")

    print("\nRun `python scripts/aggregate_results.py --study "
          f"{cfg.study}` next to fold this into summary.json and RESULTS.md.")


if __name__ == "__main__":
    main()

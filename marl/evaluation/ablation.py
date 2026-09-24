"""Input-ablation evaluation (§22.4 item 4, §24.4 criterion 4): does
zeroing one local-observation channel at evaluation change outcomes
measurably? Used both by Phase 7's "genuine learning" checklist and by
Phase 8's coordination-emergence decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from marl.algorithms.common.policy import AblatedActorPolicy, LearnedActorPolicy
from marl.analysis.metrics import compute_episode_metrics
from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.evaluation.evaluator import run_episode
from marl.evaluation.final_eval import episode_seed

# Channels ablated (§8.1): 5 = visited_known (team exploration memory),
# 6 = teammate (currently-sensed teammates).
ABLATION_CHANNELS = {"channel5_visited_known": 5, "channel6_teammate": 6}


def _eval_arm(env_cfg: EnvConfig, make_policy, maps: Sequence[MapSpec], eval_seeds: Sequence[int]) -> dict[str, float]:
    successes, collisions = [], []
    for m in maps:
        for s in eval_seeds:
            rng = np.random.default_rng(episode_seed(m.id, s))
            result = run_episode(env_cfg, make_policy(), rng, map_spec=m)
            metrics = compute_episode_metrics(result.trace, result.n_reachable_free)
            successes.append(float(metrics.success))
            collisions.append(float(metrics.n_collisions))
    return {"success_rate": float(np.mean(successes)), "collisions": float(np.mean(collisions))}


def run_ablation_for_checkpoint(
    env_cfg: EnvConfig, actor: Any, maps: Sequence[MapSpec], eval_seeds: Sequence[int]
) -> dict[str, dict[str, float]]:
    """{"baseline": {...}, "channel5_visited_known": {...}, "channel6_teammate": {...}}
    for one trained actor, evaluated stochastically over `maps` x `eval_seeds`."""
    out = {"baseline": _eval_arm(env_cfg, lambda: LearnedActorPolicy(actor, stochastic=True), maps, eval_seeds)}
    for name, channel in ABLATION_CHANNELS.items():
        out[name] = _eval_arm(
            env_cfg, lambda ch=channel: AblatedActorPolicy(actor, channel=ch, stochastic=True), maps, eval_seeds
        )
    return out


# A floor so a baseline of exactly 0 collisions doesn't produce a division
# by zero or an infinite/undefined multiplier.
_COLLISION_FLOOR = 0.02


def summarize_ablation(per_seed_results: Sequence[dict[str, dict[str, float]]]) -> dict[str, Any]:
    """Median across seeds of: the ch6 collision-rate multiplier, and the
    ch5 success-rate drop (§24.4 criterion 4's materiality thresholds
    operate on these two numbers)."""
    ch6_mult = [
        r["channel6_teammate"]["collisions"] / max(r["baseline"]["collisions"], _COLLISION_FLOOR)
        for r in per_seed_results
    ]
    ch5_drop = [r["baseline"]["success_rate"] - r["channel5_visited_known"]["success_rate"] for r in per_seed_results]
    return {
        "n_seeds": len(per_seed_results),
        "channel6_collision_multiplier_per_seed": ch6_mult,
        "channel6_collision_multiplier_median": float(np.median(ch6_mult)) if ch6_mult else None,
        "channel5_success_drop_per_seed": ch5_drop,
        "channel5_success_drop_median": float(np.median(ch5_drop)) if ch5_drop else None,
    }

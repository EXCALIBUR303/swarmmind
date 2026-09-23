"""Periodic validation evaluation during training (§21.3).

Runs the current actor (wrapped as a `Policy`) over the level's frozen
validation set, stochastically, using the SAME `run_episode`/`metrics`
code path as everything else (§5.3). Never touches the test set.
"""

from __future__ import annotations

import numpy as np

from marl.algorithms.common.policy import LearnedActorPolicy
from marl.analysis.metrics import compute_episode_metrics
from marl.envs.core.config import EnvConfig
from marl.envs.core.mapgen import generate_map
from marl.evaluation.eval_sets import VALIDATION_SEED_BASE, eval_set_exists, load_eval_set
from marl.evaluation.evaluator import run_episode
from marl.networks.actor import ActorPolicy
from marl.utils.hashing import sha256_hex


def _load_or_generate_validation_maps(env_cfg: EnvConfig, level: str, n_maps: int) -> list:
    """Prefer the frozen, committed validation set for a registered level
    (L1-L4). Falls back to generating the same deterministic seed range
    on the fly for ad hoc configs (e.g. a one-off smoke-test env) that
    were never frozen via `scripts/make_eval_maps.py` — same seeds, same
    reproducibility, just not persisted to disk (§20.3's isolation
    guarantee is about TRAINING never touching validation/test hashes;
    it doesn't require every possible config to have a frozen file)."""
    if eval_set_exists(level, "validation"):
        return load_eval_set(level, "validation")[:n_maps]
    return [generate_map(env_cfg, seed=VALIDATION_SEED_BASE + i, id_prefix=f"{level}-val") for i in range(n_maps)]


def evaluate_on_validation(
    env_cfg: EnvConfig,
    actor: ActorPolicy,
    level: str,
    n_maps: int = 50,
    stochastic: bool = True,
) -> dict[str, float | None]:
    """Return aggregate validation metrics: success_rate, mean_return,
    mean_coverage, mean_steps_on_success, censored_mean_steps."""
    was_training = actor.training
    actor.eval()
    try:
        maps = _load_or_generate_validation_maps(env_cfg, level, n_maps)
        policy = LearnedActorPolicy(actor, stochastic=stochastic)

        successes = 0
        returns = []
        coverages = []
        steps_on_success = []
        censored_steps = []

        for map_spec in maps:
            seed = int(sha256_hex(map_spec.id)[:8], 16)
            rng = np.random.default_rng(seed)
            result = run_episode(env_cfg, policy, rng, map_spec=map_spec)
            metrics = compute_episode_metrics(result.trace, result.n_reachable_free)

            returns.append(metrics.episode_return_mean)
            coverages.append(metrics.coverage_final)
            if metrics.success:
                successes += 1
                steps_on_success.append(metrics.steps_to_completion)
                censored_steps.append(metrics.steps_to_completion)
            else:
                censored_steps.append(env_cfg.max_steps)

        n = len(maps)
        return {
            "success_rate": successes / max(n, 1),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
            "mean_steps_on_success": float(np.mean(steps_on_success)) if steps_on_success else None,
            "censored_mean_steps": float(np.mean(censored_steps)) if censored_steps else None,
        }
    finally:
        if was_training:
            actor.train()

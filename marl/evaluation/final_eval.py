"""Final held-out evaluation (§22.1): the ONLY code that reads the test and
structured map sets.

Protocol (frozen in docs/EXPERIMENTS.md, Phase 6):
- every map in the split × evaluation sampling seeds {0, 1};
- common random numbers: the policy's sampling RNG for (map, eval_seed) is
  the same for every method and every training seed, so methods differ
  only by their policy, never by their luck;
- stochastic sampling is primary; greedy argmax is secondary and stored
  separately (§22.3);
- a learned run is evaluated only if its manifest says `completed`, and
  every held-out access is appended to an audit log (§20.3).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from marl.algorithms.common.policy import Policy
from marl.analysis.metrics import compute_episode_metrics
from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.evaluation.eval_sets import load_eval_set
from marl.evaluation.evaluator import run_episode
from marl.utils.hashing import sha256_hex
from marl.utils.paths import RESULTS_DIR

HELD_OUT_SPLITS = ("test", "structured")
DEFAULT_EVAL_SEEDS = (0, 1)
# Bump if the per-episode record or seeding changes: results files written
# under a different protocol version are never silently mixed.
EVAL_PROTOCOL_VERSION = 2  # 2 (Phase 15): reset-time discoveries counted; §23 workload credit
ACCESS_LOG = RESULTS_DIR / "heldout_access_log.jsonl"


class HeldOutAccessError(RuntimeError):
    """Raised when held-out maps are requested for a run that isn't finished."""


def episode_seed(map_id: str, eval_seed: int, sampling_seed: int | None = None) -> int:
    """Policy-sampling seed for one evaluation episode (common random
    numbers). `sampling_seed` distinguishes the Random baseline's 5
    evaluation "seeds" (its unit of analysis); learned policies use None,
    so every training seed and both algorithms see identical draws."""
    key = f"{map_id}:{eval_seed}" if sampling_seed is None else f"{map_id}:{eval_seed}:{sampling_seed}"
    return int(sha256_hex(key)[:8], 16)


def _episode_record(cfg: EnvConfig, map_spec: MapSpec, eval_seed: int, result) -> dict[str, Any]:
    m = compute_episode_metrics(result.trace, result.n_reachable_free)
    agent_steps = max(cfg.n_robots * m.steps_to_completion, 1)
    return {
        "map_id": map_spec.id,
        "map_hash": map_spec.content_hash,
        "eval_seed": eval_seed,
        "success": bool(m.success),
        "steps": int(m.steps_to_completion),
        "censored_steps": int(m.steps_to_completion) if m.success else int(cfg.max_steps),
        "return_mean": m.episode_return_mean,
        "reward_components": m.reward_components_total,
        "rescue_rate": m.rescue_rate,
        "discover_rate": m.discover_rate,
        "coverage": m.coverage_final,
        "visited_coverage": m.visited_coverage_final,
        "collisions": m.n_collisions,
        "collisions_per_100_agent_steps": 100.0 * m.n_collisions / agent_steps,
        "invalid_actions": m.n_invalid_actions,
        "exploration_efficiency": m.exploration_efficiency,
        "rescue_efficiency": m.rescue_efficiency,
        "rescue_latency": m.rescue_latency_mean,
        "idle_fraction": float(np.mean(m.idle_fraction_per_robot)),
        "oscillation_rate": float(np.mean(m.oscillation_rate_per_robot)),
        "workload_gini": m.workload_gini,
    }


# Per-episode fields averaged into the per-run aggregate (§23). The
# success-conditional mean (`steps_on_success`) is added separately.
AGGREGATE_FIELDS = (
    "success",
    "censored_steps",
    "return_mean",
    "rescue_rate",
    "discover_rate",
    "coverage",
    "visited_coverage",
    "collisions_per_100_agent_steps",
    "invalid_actions",
    "exploration_efficiency",
    "rescue_efficiency",
    "idle_fraction",
    "oscillation_rate",
    "workload_gini",
)


def aggregate_episodes(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("no episodes to aggregate")
    out: dict[str, Any] = {"n_episodes": len(episodes)}
    for key in AGGREGATE_FIELDS:
        out[key if key != "success" else "success_rate"] = float(np.mean([float(e[key]) for e in episodes]))
    wins = [e["steps"] for e in episodes if e["success"]]
    out["steps_on_success"] = float(np.mean(wins)) if wins else None
    lat = [e["rescue_latency"] for e in episodes if e["rescue_latency"] is not None]
    out["rescue_latency"] = float(np.mean(lat)) if lat else None
    return out


def evaluate_policy(
    cfg: EnvConfig,
    make_policy: Callable[[], Policy],
    maps: Sequence[MapSpec],
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    sampling_seed: int | None = None,
) -> dict[str, Any]:
    """Run `make_policy()` on every (map, eval_seed); return per-episode
    records plus their aggregate. A fresh policy per episode means no state
    (e.g. a heuristic's plan cache) can leak between episodes."""
    episodes = []
    for map_spec in maps:
        for s in eval_seeds:
            rng = np.random.default_rng(episode_seed(map_spec.id, s, sampling_seed))
            result = run_episode(cfg, make_policy(), rng, map_spec=map_spec)
            episodes.append(_episode_record(cfg, map_spec, s, result))
    return {"aggregate": aggregate_episodes(episodes), "episodes": episodes}


def log_heldout_access(entry: dict[str, Any], log_path: Path | None = None) -> None:
    path = log_path or ACCESS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now(UTC).isoformat(), "pid": os.getpid(), **entry}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_heldout_maps(
    level: str,
    split: str,
    requester: dict[str, Any],
    run_status: str | None = None,
    log_path: Path | None = None,
) -> list[MapSpec]:
    """The gatekeeper for test/structured maps. `run_status` is the
    evaluated run's manifest status (None for non-learned baselines)."""
    if split not in HELD_OUT_SPLITS:
        raise ValueError(f"{split!r} is not a held-out split {HELD_OUT_SPLITS}")
    if run_status is not None and run_status != "completed":
        raise HeldOutAccessError(
            f"refusing held-out '{split}' evaluation of a run with status {run_status!r}; "
            "only completed runs may touch held-out maps (§20.3)"
        )
    maps = load_eval_set(level, split)
    log_heldout_access({"level": level, "split": split, "n_maps": len(maps), **requester}, log_path)
    return maps


def write_eval_file(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write (§37)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=1, default=float)
    tmp.replace(path)


def eval_file_is_current(path: Path, expected: dict[str, Any]) -> bool:
    """True iff `path` exists and was written with exactly `expected`
    protocol fields (so a changed protocol is re-evaluated, never mixed)."""
    if not path.is_file():
        return False
    try:
        with path.open() as f:
            meta = json.load(f).get("protocol", {})
    except (OSError, json.JSONDecodeError):
        return False
    return all(meta.get(k) == v for k, v in expected.items())


def protocol_fields(level: str, split: str, eval_seeds: Sequence[int], modes: Sequence[str]) -> dict[str, Any]:
    return {
        "version": EVAL_PROTOCOL_VERSION,
        "level": level,
        "split": split,
        "eval_seeds": list(eval_seeds),
        "modes": list(modes),
    }

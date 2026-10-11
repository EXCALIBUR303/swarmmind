"""Running episodes for §24 coordination analysis: the team run, the
isolation-null run (§24.3), and per-seed aggregation with CIs (§4/§22.2's
same rules — the unit of analysis is still the seed).

Reuses `marl.evaluation.evaluator.run_episode` (§5.3 single code path) for
every episode; the only new logic here is the isolation-null map/config
construction and the aggregation across episodes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from marl.algorithms.common.policy import Policy
from marl.analysis.coordination import (
    CoordinationMetrics,
    compute_coordination_metrics,
    duplicate_exploration_ratio,
    sensed_masks_per_robot,
)
from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.sensing import sensed_mask
from marl.envs.core.types import OBSTACLE
from marl.evaluation.evaluator import run_episode
from marl.evaluation.final_eval import episode_seed
from marl.evaluation.stats import Interval, t_interval

# Fields aggregated straight (present in every episode).
_ALWAYS = ("der", "visitation_overlap", "mean_inter_robot_distance", "min_inter_robot_distance")
# Fields that can be undefined for a given episode; aggregated over the
# episodes where they are defined, with the excluded count reported.
_SOMETIMES = ("der_at_50", "redundant_pursuit", "allocation_optimality", "dispersion_speed")


def isolation_map_and_config(env_cfg: EnvConfig, map_spec: MapSpec, robot_idx: int) -> tuple[EnvConfig, MapSpec]:
    """A single-robot (N=1) config and map for the isolation null (§24.3):
    the SAME terrain and targets, but only `robot_idx`'s start cell, so
    `n_targets` and everything else about the map stays identical — only
    the presence of teammates is removed."""
    iso_cfg = env_cfg.model_copy(update={"n_robots": 1})
    iso_map = dataclasses.replace(
        map_spec, id=f"{map_spec.id}-iso{robot_idx}", robot_starts=[map_spec.robot_starts[robot_idx]]
    )
    return iso_cfg, iso_map


def run_isolation_null(
    env_cfg: EnvConfig,
    make_policy: Callable[[], Policy],
    map_spec: MapSpec,
    eval_seed: int,
    n_robots: int,
    sampling_seed: int | None = None,
) -> tuple[float, float | None]:
    """Isolation null (§24.3) for one map/eval_seed: run each of `n_robots` robots ALONE on the same
    map. Returns `(der_iso_full, der_iso_at_50)`:

    - `der_iso_full`: DER over each solo run's whole episode (descriptive only). A lone robot runs
      much longer than a team, so this is inflated by episode length (§24.1) and is NOT comparable
      with the team's DER@50.
    - `der_iso_at_50`: the matched-coverage null. The solo trajectories are played on one shared
      clock (a robot whose solo episode ended stays where it stopped) and DER is taken at the first
      step at which their combined sensing covers >= 50% of the free cells — exactly how the team's
      DER@50 is measured. `None` if that coverage is never reached. (Phase 15 fix: criterion 2b used
      to compare the team's DER@50 with `der_iso_full`, which is biased towards passing.)"""
    terrain = map_spec.terrain_array()
    free = terrain != OBSTACLE
    n_free = max(int(free.sum()), 1)
    trajectories, full_masks = [], []
    for i in range(n_robots):
        iso_cfg, iso_map = isolation_map_and_config(env_cfg, map_spec, i)
        rng = np.random.default_rng(episode_seed(f"{map_spec.id}:iso{i}", eval_seed, sampling_seed))
        result = run_episode(iso_cfg, make_policy(), rng, map_spec=iso_map)
        trajectories.append([p[0] for p in result.trace.positions])
        (mask,) = sensed_masks_per_robot(result.trace.positions, iso_cfg.sensor_radius, iso_cfg.height, iso_cfg.width)
        full_masks.append(mask)
    der_full = duplicate_exploration_ratio(full_masks)

    cumulative = [np.zeros_like(free) for _ in range(n_robots)]
    for t in range(max(len(tr) for tr in trajectories)):
        for i, tr in enumerate(trajectories):
            y, x = tr[min(t, len(tr) - 1)]
            cumulative[i] |= sensed_mask((int(y), int(x)), env_cfg.sensor_radius, env_cfg.height, env_cfg.width)
        union = np.logical_or.reduce(cumulative)
        if (union & free).sum() / n_free >= 0.5:
            return der_full, duplicate_exploration_ratio(cumulative)
    return der_full, None


def run_coordination_episode(
    env_cfg: EnvConfig,
    make_policy: Callable[[], Policy],
    map_spec: MapSpec,
    eval_seed: int,
    with_isolation: bool = True,
    sampling_seed: int | None = None,
) -> tuple[CoordinationMetrics, tuple[float, float | None] | None]:
    """One team episode's coordination metrics, plus `(der_iso_full, der_iso_at_50)` if requested."""
    rng = np.random.default_rng(episode_seed(map_spec.id, eval_seed, sampling_seed))
    result = run_episode(env_cfg, make_policy(), rng, map_spec=map_spec)
    terrain = map_spec.terrain_array()
    metrics = compute_coordination_metrics(result.trace, terrain, map_spec.targets, env_cfg.sensor_radius)
    der_iso = None
    if with_isolation and env_cfg.n_robots > 1:
        der_iso = run_isolation_null(env_cfg, make_policy, map_spec, eval_seed, env_cfg.n_robots, sampling_seed)
    return metrics, der_iso


def aggregate_coordination(
    per_episode: Sequence[CoordinationMetrics], der_iso_values: Sequence[tuple[float, float | None]]
) -> dict[str, Any]:
    """One seed's worth of episodes -> means (with counts of excluded/
    undefined episodes for the `_SOMETIMES` fields)."""
    out: dict[str, Any] = {"n_episodes": len(per_episode)}
    for field_name in _ALWAYS:
        out[field_name] = float(np.mean([getattr(m, field_name) for m in per_episode]))
    for field_name in _SOMETIMES:
        vals = [v for m in per_episode if (v := getattr(m, field_name)) is not None]
        out[field_name] = {"mean": float(np.mean(vals)) if vals else None, "n_defined": len(vals),
                           "n_excluded": len(per_episode) - len(vals)}
    full = [v[0] for v in der_iso_values]
    at50 = [v[1] for v in der_iso_values if v[1] is not None]
    out["der_iso"] = float(np.mean(full)) if full else None  # whole solo episodes: descriptive only
    out["der_iso_at_50"] = (
        {"mean": float(np.mean(at50)) if at50 else None, "n_defined": len(at50), "n_excluded": len(full) - len(at50)}
        if der_iso_values else None
    )
    return out


def per_seed_interval(per_seed_means: Sequence[float]) -> Interval:
    return t_interval(list(per_seed_means))


def run_coordination_for_seed(
    env_cfg: EnvConfig,
    make_policy: Callable[[], Policy],
    maps: Sequence[MapSpec],
    eval_seeds: Sequence[int],
    with_isolation: bool = True,
    sampling_seed: int | None = None,
) -> dict[str, Any]:
    """One (algorithm, seed)'s coordination aggregate over `maps` x
    `eval_seeds` (common random numbers, same as `final_eval` — same
    (map, eval_seed) pairing every method and seed gets)."""
    per_episode: list[CoordinationMetrics] = []
    der_iso_values: list[tuple[float, float | None]] = []
    for map_spec in maps:
        for s in eval_seeds:
            metrics, der_iso = run_coordination_episode(env_cfg, make_policy, map_spec, s, with_isolation, sampling_seed)
            per_episode.append(metrics)
            if der_iso is not None:
                der_iso_values.append(der_iso)
    return aggregate_coordination(per_episode, der_iso_values)

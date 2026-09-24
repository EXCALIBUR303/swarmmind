"""Behaviour analysis (§26): per-episode curves and per-map heatmaps as
JSON, produced from the same `EpisodeTrace`/`MapSpec` data every other
analysis module uses (§5.3 single code path). The frontend renders the
interactive versions from this same JSON (§26); the Python side also
turns it into static figures for the written report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from marl.analysis.coordination import duplicate_exploration_ratio
from marl.analysis.metrics import compute_episode_metrics
from marl.envs.core.types import OBSTACLE, EpisodeTrace


def der_over_time(positions: list[np.ndarray], sensor_radius: float, height: int, width: int) -> list[float]:
    """DER at every step t = 0..T_ep, using the cumulative sensed masks up
    to and including t (§24.1's "DER vs. coverage curve", generalized to
    every step rather than only the coverage=50% point)."""
    n = positions[0].shape[0]
    masks = [np.zeros((height, width), dtype=bool) for _ in range(n)]
    curve = []
    for pos_t in positions:
        for i in range(n):
            masks[i] |= _sensed_mask_at(pos_t[i], sensor_radius, height, width)
        curve.append(duplicate_exploration_ratio(masks))
    return curve


def _sensed_mask_at(pos: np.ndarray, sensor_radius: float, height: int, width: int) -> np.ndarray:
    from marl.envs.core.sensing import sensed_mask

    return sensed_mask((int(pos[0]), int(pos[1])), sensor_radius, height, width)


def inter_robot_distance_over_time(positions: list[np.ndarray]) -> list[dict[str, float]]:
    """Per-step {mean, min} pairwise inter-robot distance (§26's "Inter-robot
    distance" line chart). Empty list for a single robot."""
    import math

    n = positions[0].shape[0]
    if n < 2:
        return []
    out = []
    for pos_t in positions:
        dists = [math.dist(pos_t[i], pos_t[j]) for i in range(n) for j in range(i + 1, n)]
        out.append({"mean": float(np.mean(dists)), "min": float(np.min(dists))})
    return out


def discovery_timeline(trace: EpisodeTrace) -> list[dict[str, Any]]:
    """Per-target {target, t_discovered, t_rescued, rescued_by} (§26's
    Gantt-style discovery/rescue timeline). `positions`-indexed (index 0 =
    the reset state), matching every other module's convention. A target
    never discovered/rescued this episode has `None` for that field."""
    discovered_at: dict[int, int] = {}
    for i, ks in enumerate(trace.discovered_events):
        for k in ks:
            discovered_at[k] = i + 1
    rescued_at: dict[int, tuple[int, int]] = {}
    for i, events in enumerate(trace.rescued_events):
        for k, robot in events:
            rescued_at[k] = (i + 1, robot)

    targets = sorted(set(discovered_at) | set(rescued_at) | set(range(trace.n_targets)))
    out = []
    for k in targets:
        r = rescued_at.get(k)
        out.append({
            "target": k,
            "t_discovered": discovered_at.get(k),
            "t_rescued": r[0] if r else None,
            "rescued_by": r[1] if r else None,
        })
    return out


def episode_behaviour(
    trace: EpisodeTrace, n_reachable_free: int, sensor_radius: float
) -> dict[str, Any]:
    """One episode's full §26 behaviour bundle (everything EXCEPT the
    heatmaps, which need many episodes on one map — see `accumulate_heatmaps`)."""
    metrics = compute_episode_metrics(trace, n_reachable_free)
    return {
        "map_id": trace.map_id,
        "success": trace.success,
        "final_t": trace.final_t,
        "coverage_over_time": list(trace.coverage_over_time),
        "der_over_time": der_over_time(trace.positions, sensor_radius, trace.height, trace.width),
        "inter_robot_distance_over_time": inter_robot_distance_over_time(trace.positions),
        "discovery_timeline": discovery_timeline(trace),
        "workload_share_per_robot": metrics.workload_share_per_robot,
        "workload_gini": metrics.workload_gini,
        "idle_fraction_per_robot": metrics.idle_fraction_per_robot,
        "distance_travelled_per_robot": metrics.distance_travelled_per_robot,
    }


def accumulate_heatmaps(
    traces: list[EpisodeTrace], height: int, width: int
) -> dict[str, list[list[int]]]:
    """Visit-count and collision-count heatmaps (§26), aggregated over
    several episodes played on the SAME map. `visit_counts[y][x]` = number
    of (episode, robot, step) triples that occupied (y,x); `collision_counts`
    counts `blocked_by_robot` events by the blocked robot's cell."""
    visit = np.zeros((height, width), dtype=np.int64)
    collide = np.zeros((height, width), dtype=np.int64)
    for trace in traces:
        for pos_t in trace.positions:
            for y, x in pos_t:
                visit[int(y), int(x)] += 1
        for pos_t, blocked in zip(trace.positions[1:], trace.blocked_by_robot, strict=True):
            for i, was_blocked in enumerate(blocked):
                if was_blocked:
                    y, x = pos_t[i]
                    collide[int(y), int(x)] += 1
    return {"visit_counts": visit.tolist(), "collision_counts": collide.tolist()}


def obstacle_mask(terrain: np.ndarray) -> list[list[bool]]:
    """For overlaying heatmaps on the map layout in the frontend/figures."""
    return (terrain == OBSTACLE).tolist()

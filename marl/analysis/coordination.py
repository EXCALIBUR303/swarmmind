"""Coordination metrics (§24), computed from an `EpisodeTrace` plus the
`MapSpec` it was played on (for terrain and target positions — the trace
itself carries only `map_id`).

Every metric here is a pure function of already-recorded data:
`trace.positions[t]` (int32[N,2], t = 0..T_ep, index 0 = the reset state,
index t = the state after t steps — the SAME indexing `trace.coverage_over_time`
uses) and the per-step lists (`discovered_events[i]`/`rescued_events[i]`,
i = 0..T_ep-1, describing step i+1; the resulting state is `positions[i+1]`).
Re-deriving each robot's own sensed cells from its position trajectory
(rather than storing them in the trace) keeps `EpisodeTrace` unchanged and
works because sensing has no occlusion and no memory beyond position
(§11.2): `sensed(i)` at any step depends only on robot i's position at
that step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from marl.envs.core.mapgen import bfs_distances
from marl.envs.core.sensing import sensed_mask
from marl.envs.core.types import OBSTACLE, EpisodeTrace


@dataclass
class CoordinationMetrics:
    """One episode's §24 coordination metrics. `None` marks an undefined
    quantity for this episode (e.g. coverage never reached 50%) — callers
    aggregate over only the episodes where a given field is defined, and
    report how many were excluded."""

    der: float
    der_at_50: float | None
    visitation_overlap: float
    redundant_pursuit: float | None  # mean over targets that were both discovered and rescued
    allocation_optimality: float | None  # fraction of such targets rescued by the BFS-nearest robot at discovery
    mean_inter_robot_distance: float
    min_inter_robot_distance: float
    dispersion_speed: int | None  # steps until mean inter-robot distance first exceeds sensor_radius


def sensed_masks_per_robot(
    positions: list[np.ndarray], sensor_radius: float, height: int, width: int, up_to: int | None = None
) -> list[np.ndarray]:
    """Per-robot cumulative sensed mask (bool[H,W]), OR-ed over
    `positions[0:up_to+1]` (default: the whole episode)."""
    n = positions[0].shape[0]
    end = len(positions) - 1 if up_to is None else up_to
    masks = [np.zeros((height, width), dtype=bool) for _ in range(n)]
    for t in range(end + 1):
        pos_t = positions[t]
        for i in range(n):
            masks[i] |= sensed_mask((int(pos_t[i, 0]), int(pos_t[i, 1])), sensor_radius, height, width)
    return masks


def duplicate_exploration_ratio(sensed_masks: list[np.ndarray]) -> float:
    """DER = 1 - |union S_i| / sum |S_i| (§24.1). 0 with no overlap, up to
    (N-1)/N when every robot senses the same cells. Robots that sensed
    nothing (N=1 with no cells sensed can't happen given sensor_radius>=1,
    but guard anyway) contribute 0 to both terms."""
    sizes = [int(m.sum()) for m in sensed_masks]
    total = sum(sizes)
    if total == 0:
        return 0.0
    union = np.zeros_like(sensed_masks[0])
    for m in sensed_masks:
        union |= m
    return 1.0 - int(union.sum()) / total


def _first_step_at_coverage(coverage_over_time: list[float], threshold: float) -> int | None:
    for t, c in enumerate(coverage_over_time):
        if c >= threshold:
            return t
    return None


def _visited_cells_per_robot(positions: list[np.ndarray]) -> list[set[tuple[int, int]]]:
    n = positions[0].shape[0]
    visited: list[set[tuple[int, int]]] = [set() for _ in range(n)]
    for pos_t in positions:
        for i in range(n):
            visited[i].add((int(pos_t[i, 0]), int(pos_t[i, 1])))
    return visited


def visitation_overlap(positions: list[np.ndarray]) -> float:
    """|{cells visited by >= 2 robots}| / |union of visited cells| (§24.2).
    Lower means better division of the map among robots."""
    visited = _visited_cells_per_robot(positions)
    counts: dict[tuple[int, int], int] = {}
    for s in visited:
        for c in s:
            counts[c] = counts.get(c, 0) + 1
    union = len(counts)
    if union == 0:
        return 0.0
    shared = sum(1 for v in counts.values() if v >= 2)
    return shared / union


def inter_robot_distances(positions: list[np.ndarray]) -> tuple[float, float]:
    """(mean, min) of the pairwise Euclidean inter-robot distance, averaged
    (mean) / taken (min) over the whole episode (§24.2). Undefined (returns
    (0.0, 0.0)) for a single robot, since there are no pairs."""
    n = positions[0].shape[0]
    if n < 2:
        return 0.0, 0.0
    means, mins = [], []
    for pos_t in positions:
        dists = [
            math.dist(pos_t[i], pos_t[j]) for i in range(n) for j in range(i + 1, n)
        ]
        means.append(float(np.mean(dists)))
        mins.append(float(np.min(dists)))
    return float(np.mean(means)), float(np.min(mins))


def dispersion_speed(positions: list[np.ndarray], sensor_radius: float) -> int | None:
    """First step index (matching `positions`' own indexing, so 0 = the
    initial state) at which the mean inter-robot distance exceeds
    `sensor_radius` (§24.2, r_s = sensor_radius per §11.1). `None` if it
    never does, or there are fewer than 2 robots."""
    n = positions[0].shape[0]
    if n < 2:
        return None
    for t, pos_t in enumerate(positions):
        dists = [math.dist(pos_t[i], pos_t[j]) for i in range(n) for j in range(i + 1, n)]
        if float(np.mean(dists)) > sensor_radius:
            return t
    return None


def _target_windows(
    trace: EpisodeTrace,
) -> dict[int, tuple[int, int, int]]:
    """For every target discovered AND rescued this episode:
    `{k: (discovered_step_idx, rescued_step_idx, rescuing_robot)}`, using
    `positions`' indexing (i.e. per-step-list index i -> positions[i+1])."""
    discovered_at: dict[int, int] = {k: 0 for k in trace.initial_discovered}  # sensed at reset: positions[0]
    for i, ks in enumerate(trace.discovered_events):
        for k in ks:
            discovered_at[k] = i + 1
    windows: dict[int, tuple[int, int, int]] = {}
    for i, events in enumerate(trace.rescued_events):
        for k, robot in events:
            if k in discovered_at:
                windows[k] = (discovered_at[k], i + 1, robot)
    return windows


def redundant_pursuit_and_allocation(
    trace: EpisodeTrace, free: np.ndarray, targets: list[tuple[int, int]]
) -> tuple[float | None, float | None]:
    """(redundant_pursuit, allocation_optimality) over targets discovered
    AND rescued this episode (§24.2). `None` if there are none, or every
    window has zero length (rescued the same step it was discovered)."""
    windows = _target_windows(trace)
    if not windows:
        return None, None

    pursuit_samples: list[int] = []
    optimal = 0
    n_scored = 0
    for k, (d, r, rescuer) in windows.items():
        dist_k = bfs_distances(free, targets[k])
        # Allocation optimality: was the rescuer the BFS-nearest robot at
        # the moment of discovery (position d)?
        pos_d = trace.positions[d]
        dists_at_discovery = [dist_k[int(pos_d[i, 0]), int(pos_d[i, 1])] for i in range(pos_d.shape[0])]
        nearest = int(np.argmin(dists_at_discovery))
        optimal += int(nearest == rescuer)
        n_scored += 1
        # Redundant pursuit: for each step in [d, r), how many robots'
        # true BFS distance to k strictly decreased over that step.
        for t in range(d, r):
            before = trace.positions[t]
            after = trace.positions[t + 1]
            n_closing = sum(
                1
                for i in range(before.shape[0])
                if dist_k[int(after[i, 0]), int(after[i, 1])] < dist_k[int(before[i, 0]), int(before[i, 1])]
            )
            pursuit_samples.append(n_closing)

    allocation = optimal / n_scored if n_scored else None
    pursuit = float(np.mean(pursuit_samples)) if pursuit_samples else None
    return pursuit, allocation


def compute_coordination_metrics(
    trace: EpisodeTrace, terrain: np.ndarray, targets: list[tuple[int, int]], sensor_radius: float
) -> CoordinationMetrics:
    """All of §24's per-episode coordination metrics from one trace."""
    free = terrain != OBSTACLE
    height, width = terrain.shape

    full_masks = sensed_masks_per_robot(trace.positions, sensor_radius, height, width)
    der = duplicate_exploration_ratio(full_masks)

    t50 = _first_step_at_coverage(trace.coverage_over_time, 0.5)
    der50 = None
    if t50 is not None:
        masks50 = sensed_masks_per_robot(trace.positions, sensor_radius, height, width, up_to=t50)
        der50 = duplicate_exploration_ratio(masks50)

    overlap = visitation_overlap(trace.positions)
    mean_dist, min_dist = inter_robot_distances(trace.positions)
    disp = dispersion_speed(trace.positions, sensor_radius)
    pursuit, allocation = redundant_pursuit_and_allocation(trace, free, targets)

    return CoordinationMetrics(
        der=der,
        der_at_50=der50,
        visitation_overlap=overlap,
        redundant_pursuit=pursuit,
        allocation_optimality=allocation,
        mean_inter_robot_distance=mean_dist,
        min_inter_robot_distance=min_dist,
        dispersion_speed=disp,
    )

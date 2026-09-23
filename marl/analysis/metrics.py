"""Episode metrics computed from a standard `EpisodeTrace` (§23).

Every consumer (evaluation, backend, replay analysis) computes metrics
through this module so that reward, metrics, and displayed numbers always
agree — the single-code-path principle (§5.3) applied to analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marl.envs.core.types import EpisodeTrace


@dataclass
class EpisodeMetrics:
    """One episode's worth of metrics (§23). All fields are plain floats/
    ints so this serializes trivially to JSON."""

    episode_return_mean: float  # mean over agents of total return
    reward_components_total: dict[str, float]  # summed over the episode, mean over agents
    success: bool
    n_discovered: int
    n_rescued: int
    n_targets: int
    discover_rate: float
    rescue_rate: float
    coverage_final: float
    visited_coverage_final: float
    steps_to_completion: int  # T_ep if successful; else censored at max_steps (caller decides which max_steps to use)
    n_collisions: int  # sum of blocked_by_robot events (agent-steps)
    n_invalid_actions: int  # sum of bumped_wall events (agent-steps); 0 by construction when masking is on
    distance_travelled_total: float
    distance_travelled_per_robot: list[float]
    idle_fraction_per_robot: list[float]
    oscillation_rate_per_robot: list[float]
    exploration_efficiency: float  # newly-known free cells per agent-step
    rescue_efficiency: float  # targets rescued per agent-step
    rescue_latency_mean: float | None  # mean(t_rescued - t_discovered) over rescued targets; None if none rescued
    workload_share_per_robot: list[float] = field(default_factory=list)
    workload_gini: float = 0.0


def compute_episode_metrics(trace: EpisodeTrace, n_reachable_free: int) -> EpisodeMetrics:
    """Compute all §23 metrics from a completed `EpisodeTrace`."""
    n = trace.n_robots
    t_ep = len(trace.actions)

    if t_ep == 0:
        # Degenerate zero-step trace (shouldn't normally occur); return
        # all-zero metrics rather than crashing on empty arrays.
        return EpisodeMetrics(
            episode_return_mean=0.0,
            reward_components_total={},
            success=trace.success,
            n_discovered=0,
            n_rescued=0,
            n_targets=trace.n_targets,
            discover_rate=0.0,
            rescue_rate=0.0,
            coverage_final=trace.coverage_over_time[-1] if trace.coverage_over_time else 0.0,
            visited_coverage_final=0.0,
            steps_to_completion=trace.final_t,
            n_collisions=0,
            n_invalid_actions=0,
            distance_travelled_total=0.0,
            distance_travelled_per_robot=[0.0] * n,
            idle_fraction_per_robot=[1.0] * n,
            oscillation_rate_per_robot=[0.0] * n,
            exploration_efficiency=0.0,
            rescue_efficiency=0.0,
            rescue_latency_mean=None,
        )

    rewards = np.stack(trace.rewards)  # [T, N]
    episode_return_mean = float(np.sum(rewards, axis=0).mean())

    component_totals: dict[str, float] = {}
    for step_components in trace.reward_components:
        for key, arr in step_components.items():
            component_totals[key] = component_totals.get(key, 0.0) + float(np.mean(arr))

    discovered_indices: set[int] = set()
    for d in trace.discovered_events:
        discovered_indices.update(d)
    n_discovered = len(discovered_indices)
    n_rescued = sum(len(r) for r in trace.rescued_events)

    coverage_final = trace.coverage_over_time[-1] if trace.coverage_over_time else 0.0

    positions = np.stack(trace.positions)  # [T+1, N, 2]
    visited_cells: set[tuple[int, int]] = set()
    for p in positions:
        for robot_pos in p:
            visited_cells.add((int(robot_pos[0]), int(robot_pos[1])))
    visited_coverage_final = len(visited_cells) / max(n_reachable_free, 1)

    n_collisions = int(np.sum(np.stack(trace.blocked_by_robot)))
    n_invalid_actions = int(np.sum(np.stack(trace.bumped_wall)))

    # Distance travelled: count of steps where a robot's position changed.
    diffs = positions[1:] - positions[:-1]  # [T, N, 2]
    moved_mask = np.any(diffs != 0, axis=2)  # [T, N]
    distance_per_robot = moved_mask.sum(axis=0).astype(float)  # [N]
    distance_total = float(distance_per_robot.sum())
    idle_fraction_per_robot = (1.0 - distance_per_robot / max(t_ep, 1)).tolist()

    # Oscillation: p[t+1] == p[t-1] != p[t] (immediate back-and-forth),
    # counted as a fraction of MOVED steps for that robot.
    oscillation_counts = np.zeros(n)
    for i in range(n):
        moves = 0
        oscillations = 0
        for t in range(1, t_ep):
            if not np.array_equal(positions[t, i], positions[t - 1, i]):
                moves += 1
                if t + 1 <= t_ep and np.array_equal(positions[t + 1, i], positions[t - 1, i]) and not np.array_equal(
                    positions[t + 1, i], positions[t, i]
                ):
                    oscillations += 1
        oscillation_counts[i] = oscillations / moves if moves > 0 else 0.0
    oscillation_rate_per_robot = oscillation_counts.tolist()

    exploration_efficiency = (
        (coverage_final * n_reachable_free) - (trace.coverage_over_time[0] * n_reachable_free)
    ) / max(n * t_ep, 1)
    rescue_efficiency = n_rescued / max(n * t_ep, 1)

    # Rescue latency needs t_discovered/t_rescued, which aren't stored per
    # step in EpisodeTrace directly — reconstruct from the event lists.
    discovered_at: dict[int, int] = {}
    for t, targets in enumerate(trace.discovered_events):
        for k in targets:
            discovered_at[k] = t
    rescue_latencies = []
    for t, events in enumerate(trace.rescued_events):
        for k, _robot in events:
            if k in discovered_at:
                rescue_latencies.append(t - discovered_at[k])
    rescue_latency_mean = float(np.mean(rescue_latencies)) if rescue_latencies else None

    # Workload share: credit each robot for cells it was the sole (or a
    # tied) discoverer of, i.e. present at the moment of first team-level
    # discovery of that cell — approximated here via each robot's own
    # distance travelled as a coverage-contribution proxy is insufficient;
    # instead we credit exploration by attributing each step's team
    # newly-known cells equally among robots that moved that step (a
    # reasonable, auditable approximation given EpisodeTrace's granularity).
    workload_credit = np.zeros(n)
    for t in range(t_ep):
        moved_this_step = moved_mask[t]
        n_moved = int(moved_this_step.sum())
        if n_moved == 0:
            continue
        explore_components = trace.reward_components[t].get("explore")
        if explore_components is None:
            continue
        team_explore = float(np.mean(explore_components))
        if team_explore <= 0:
            continue
        share = team_explore / n_moved
        workload_credit[moved_this_step] += share
    total_credit = workload_credit.sum()
    workload_share = (workload_credit / total_credit).tolist() if total_credit > 0 else [1.0 / n] * n
    workload_gini = _gini(np.array(workload_share))

    return EpisodeMetrics(
        episode_return_mean=episode_return_mean,
        reward_components_total=component_totals,
        success=trace.success,
        n_discovered=n_discovered,
        n_rescued=n_rescued,
        n_targets=trace.n_targets,
        discover_rate=n_discovered / max(trace.n_targets, 1),
        rescue_rate=n_rescued / max(trace.n_targets, 1),
        coverage_final=coverage_final,
        visited_coverage_final=visited_coverage_final,
        steps_to_completion=trace.final_t,
        n_collisions=n_collisions,
        n_invalid_actions=n_invalid_actions,
        distance_travelled_total=distance_total,
        distance_travelled_per_robot=distance_per_robot.tolist(),
        idle_fraction_per_robot=idle_fraction_per_robot,
        oscillation_rate_per_robot=oscillation_rate_per_robot,
        exploration_efficiency=exploration_efficiency,
        rescue_efficiency=rescue_efficiency,
        rescue_latency_mean=rescue_latency_mean,
        workload_share_per_robot=workload_share,
        workload_gini=workload_gini,
    )


def _gini(shares: np.ndarray) -> float:
    """Gini coefficient of a share distribution (§23 workload Gini)."""
    n = len(shares)
    if n == 0:
        return 0.0
    sorted_shares = np.sort(shares)
    cum = np.cumsum(sorted_shares)
    total = cum[-1]
    if total <= 0:
        return 0.0
    numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_shares)
    return float(numerator / (n * total))

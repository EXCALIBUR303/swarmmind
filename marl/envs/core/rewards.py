"""Reward function (§13.2) with the exploration-budget terminal payout (§13.3).

All positive mission rewards (explore, discover, rescue, complete) are
shared across the team (scaled by `team_share`, default 1.0 = fully
shared). The collision/bump penalty is always individual (§13.1).

The exploration budget invariant (§13.3): the total exploration reward of
ANY successful episode equals `w_x * (|G| - |K_0|)` exactly, because on
success we pay out every still-unknown cell at once. This removes the
incentive to delay the final rescue to farm exploration reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.envs.core.config import RewardConfig


@dataclass
class RewardInputs:
    n_robots: int
    n_reachable_free: int  # |G|, precomputed once per map (reachable free-cell count)
    newly_known_count: int  # |K_{t+1}| - |K_t| this step (team, first-time-known cells)
    n_newly_discovered: int  # targets newly discovered this step
    n_rescued_this_step: int  # targets rescued this step
    success: bool  # mission complete this step
    known_count_after: int  # |K_{t+1}| ∩ free, i.e. team-known free cells AFTER this step
    blocked_by_robot: np.ndarray  # bool[N] — this robot was blocked by another robot this step
    bumped_wall: np.ndarray  # bool[N]


def compute_rewards(cfg: RewardConfig, inputs: RewardInputs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute per-robot rewards and their components for one step.

    Returns (reward[N] float32, components: dict[str, float32[N]]).
    Components: "explore", "discover", "rescue", "complete", "time",
    "collision" (individual, negative already applied as a cost, stored as
    the magnitude subtracted — i.e. component values are POSITIVE
    contributions or POSITIVE costs; the caller sums with the correct sign
    as coded below).
    """
    n = inputs.n_robots
    w_x = cfg.explore_total / max(inputs.n_reachable_free, 1)

    explore_component = np.full(n, w_x * inputs.newly_known_count, dtype=np.float32)
    discover_component = np.full(n, cfg.discover * inputs.n_newly_discovered, dtype=np.float32)
    rescue_component = np.full(n, cfg.rescue * inputs.n_rescued_this_step, dtype=np.float32)

    complete_component = np.zeros(n, dtype=np.float32)
    if inputs.success:
        # §13.2's success term is ONE bracket: w_c + w_x·(|G|-|K_{t+1}|).
        # Split for reporting: the constant bonus w_c goes to "complete",
        # and the exploration payout goes to "explore" — so that the
        # exploration-budget invariant (§13.3) holds by summing "explore"
        # alone. Each piece must appear in EXACTLY ONE component, since
        # `reward` below sums all components: putting the payout in both
        # would silently double it.
        remaining_unknown = max(inputs.n_reachable_free - inputs.known_count_after, 0)
        complete_component[:] = cfg.complete
        explore_component[:] += w_x * remaining_unknown

    # Apply team_share: shared positive rewards are what they are (already
    # identical per-robot); team_share < 1.0 would mix in an individual
    # split, but v1 defaults to 1.0 (fully shared) and no per-robot credit
    # source exists yet, so team_share is applied as a uniform scale here.
    beta = cfg.team_share
    shared_positive = (
        explore_component + discover_component + rescue_component + complete_component
    ) * beta

    time_component = np.full(n, cfg.time_cost, dtype=np.float32)
    collision_component = np.where(
        inputs.blocked_by_robot | inputs.bumped_wall, cfg.collision, 0.0
    ).astype(np.float32)

    reward = shared_positive - time_component - collision_component

    components = {
        "explore": explore_component,
        "discover": discover_component,
        "rescue": rescue_component,
        "complete": complete_component,
        "time": time_component,
        "collision": collision_component,
    }
    return reward.astype(np.float32), components

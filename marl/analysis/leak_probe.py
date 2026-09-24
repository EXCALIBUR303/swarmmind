"""Statistical information-leak probe: pre-discovery target-direction bias (§12.4).

The counterfactual unit tests (§12.4) prove the actor's OBSERVATION cannot
contain hidden-target information. This probe checks the trained
BEHAVIOUR end to end: if privileged information reached the actor by any
route (a bug in observation building, a shared layer, a mis-wired input),
robots would move toward a target measurably more often while it is still
hidden than toward an equivalent cell they have no reason to prefer.

Design (paired, so exploration drive cancels out):
  - For every real target k, sample a PHANTOM cell under the same rules the
    map generator used for real targets (free, not a robot start, not a
    target, BFS distance >= target_min_dist from every start) and also not
    already known to the team at t=0.
  - For every robot-step where BOTH target k and its phantom are still
    unknown to the team, record whether the robot's move reduced its true
    BFS distance to k, and to the phantom.
  - The paired difference d = toward_real - toward_phantom ∈ {-1, 0, 1} has
    mean ≈ 0 for a leak-free policy. One-sided z-test on mean(d).

A result is FLAGGED only if the bias is both material (> `min_diff`) and
statistically significant (p < `alpha`) — with thousands of samples,
significance alone would flag negligible, meaningless differences.

Calibration (L1, 50 frozen validation maps, seed 0 — Phase 5):
  random             diff -0.004  p 0.68   not flagged
  frontier heuristic diff -0.008  p 0.65   not flagged
  oracle cheater     diff +0.265  p 2e-27  flagged   (steers to hidden targets 100% of steps)
  partial cheater    diff +0.046  p 6e-4   flagged   (does so on 30% of steps)
Leak-free policies sit within ±0.01 of zero, so `min_diff=0.03` leaves a
3x noise margin while still catching a 30%-of-the-time leak (a 0.05
margin, tried first, missed it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from marl.algorithms.common.policy import Policy
from marl.envs.core.config import EnvConfig
from marl.envs.core.mapgen import bfs_distances
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.types import HIDDEN, OBSTACLE
from marl.envs.core.world import SwarmWorld


@dataclass
class LeakProbeResult:
    n_pairs: int
    frac_toward_real: float
    frac_toward_phantom: float
    diff: float  # frac_toward_real - frac_toward_phantom
    z: float
    p_value: float  # one-sided: H1 = moves toward real hidden targets MORE often
    flagged: bool


def _sample_phantom(
    world: SwarmWorld, env_cfg: EnvConfig, rng: np.random.Generator, exclude: set[tuple[int, int]]
) -> tuple[int, int] | None:
    state = world.state
    assert state is not None
    free = state.terrain != OBSTACLE
    start_dists = [bfs_distances(free, (int(p[0]), int(p[1]))) for p in state.robot_pos]
    candidates = []
    for y, x in zip(*np.where(free), strict=True):
        cell = (int(y), int(x))
        if cell in exclude or state.known_team[cell]:
            continue
        if all(d[cell] >= env_cfg.target_min_dist for d in start_dists):
            candidates.append(cell)
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def run_leak_probe(
    env_cfg: EnvConfig,
    policy: Policy,
    maps: list[MapSpec],
    seed: int = 0,
    min_diff: float = 0.03,
    alpha: float = 0.01,
) -> LeakProbeResult:
    rng = np.random.default_rng(seed)
    diffs: list[int] = []
    toward_real_total = 0
    toward_phantom_total = 0

    for map_spec in maps:
        world = SwarmWorld(env_cfg)
        state = world.reset(map_spec=map_spec)
        free = state.terrain != OBSTACLE

        targets = [(int(p[0]), int(p[1])) for p in state.target_pos]
        exclude = set(targets) | {(int(p[0]), int(p[1])) for p in state.robot_pos}
        pairs: list[tuple[int, np.ndarray, tuple[int, int], np.ndarray]] = []
        for k, t_cell in enumerate(targets):
            phantom = _sample_phantom(world, env_cfg, rng, exclude)
            if phantom is None:
                continue
            exclude.add(phantom)
            pairs.append((k, bfs_distances(free, t_cell), phantom, bfs_distances(free, phantom)))

        episode_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        for _ in range(env_cfg.max_steps):
            before = state.robot_pos.copy()
            active = [
                (k, d_real, ph, d_ph)
                for k, d_real, ph, d_ph in pairs
                if state.target_status[k] == HIDDEN and not state.known_team[ph]
            ]
            actions = policy.act(world, episode_rng)
            result = world.step(actions)
            after = state.robot_pos
            for _k, d_real, _ph, d_ph in active:
                for i in range(state.n_robots):
                    b = (int(before[i, 0]), int(before[i, 1]))
                    a = (int(after[i, 0]), int(after[i, 1]))
                    toward_real = int(d_real[a] < d_real[b])
                    toward_ph = int(d_ph[a] < d_ph[b])
                    toward_real_total += toward_real
                    toward_phantom_total += toward_ph
                    diffs.append(toward_real - toward_ph)
            if result.terminated or result.truncated:
                break

    n = len(diffs)
    if n == 0:
        return LeakProbeResult(0, 0.0, 0.0, 0.0, 0.0, 1.0, False)
    d = np.asarray(diffs, dtype=np.float64)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = mean / se if se > 0 else 0.0
    p_value = 0.5 * math.erfc(z / math.sqrt(2.0))  # one-sided upper tail
    return LeakProbeResult(
        n_pairs=n,
        frac_toward_real=toward_real_total / n,
        frac_toward_phantom=toward_phantom_total / n,
        diff=mean,
        z=z,
        p_value=p_value,
        flagged=(mean > min_diff) and (p_value < alpha),
    )

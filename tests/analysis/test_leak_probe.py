"""Leak-probe tests (§12.4): the probe must stay quiet on leak-free policies
and fire on policies that use hidden-target information. Deterministic
given the seed, so these are exact regression checks, not flaky
statistical tests."""

from __future__ import annotations

import numpy as np
import pytest

from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy
from marl.algorithms.baselines.random_policy import RandomPolicy
from marl.analysis.leak_probe import run_leak_probe
from marl.envs.core.config import LevelRegistry
from marl.envs.core.mapgen import bfs_distances
from marl.envs.core.types import ACTION_DELTAS, HIDDEN, OBSTACLE
from marl.evaluation.eval_sets import load_eval_set


class CheatingPolicy:
    """Steers each robot toward the nearest HIDDEN target using world truth
    on a fraction `p_cheat` of steps — a leak by construction."""

    def __init__(self, p_cheat: float) -> None:
        self.p_cheat = p_cheat

    def act(self, world, rng: np.random.Generator) -> np.ndarray:
        s = world.state
        free = s.terrain != OBSTACLE
        _, _, masks = world.all_actor_obs()
        hidden = [
            (int(s.target_pos[k, 0]), int(s.target_pos[k, 1]))
            for k in range(s.n_targets)
            if s.target_status[k] == HIDDEN
        ]
        dists = [bfs_distances(free, t) for t in hidden]
        actions = np.zeros(s.n_robots, dtype=np.int32)
        for i in range(s.n_robots):
            valid = np.where(masks[i] == 1)[0]
            y, x = int(s.robot_pos[i, 0]), int(s.robot_pos[i, 1])
            best = None
            if dists and rng.random() < self.p_cheat:
                for a in valid:
                    dy, dx = ACTION_DELTAS[a]
                    cell = (y + dy, x + dx)
                    score = min(d[cell] for d in dists if d[cell] >= 0)
                    if best is None or score < best[0]:
                        best = (score, a)
            actions[i] = best[1] if best is not None else rng.choice(valid)
        return actions


@pytest.fixture(scope="module")
def l1():
    return LevelRegistry().get("L1"), load_eval_set("L1", "validation")


@pytest.mark.parametrize("policy", [RandomPolicy(), FrontierHeuristicPolicy()], ids=["random", "heuristic"])
def test_probe_not_flagged_for_leak_free_policies(l1, policy):
    cfg, maps = l1
    r = run_leak_probe(cfg, policy, maps, seed=0)
    assert r.n_pairs > 100
    assert abs(r.diff) < 0.02
    assert not r.flagged


def test_probe_flags_oracle_cheater(l1):
    cfg, maps = l1
    r = run_leak_probe(cfg, CheatingPolicy(1.0), maps, seed=0)
    assert r.flagged
    assert r.diff > 0.2


def test_probe_flags_partial_cheater(l1):
    """A policy that uses hidden-target information on only 30% of steps is
    still a leak and must still be caught at the default thresholds."""
    cfg, maps = l1
    r = run_leak_probe(cfg, CheatingPolicy(0.3), maps, seed=0)
    assert r.flagged


def test_probe_deterministic_given_seed(l1):
    cfg, maps = l1
    a = run_leak_probe(cfg, RandomPolicy(), maps[:10], seed=3)
    b = run_leak_probe(cfg, RandomPolicy(), maps[:10], seed=3)
    assert a == b

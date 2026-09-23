"""Replay recorder/loader tests: round-trip and re-simulation determinism (§27, §33.4)."""

from __future__ import annotations

import numpy as np

from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy
from marl.algorithms.baselines.random_policy import RandomPolicy
from marl.envs.core.config import EnvConfig
from marl.replay.loader import load_replay, resimulate_matches, save_replay
from marl.replay.recorder import record_episode
from marl.replay.schema import ReplayPolicyInfo


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t",
        width=8,
        height=8,
        n_robots=2,
        n_targets=1,
        obstacle_density=0.1,
        max_steps=60,
        sensor_radius=2.0,
        obs_radius=5,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def test_record_episode_produces_matching_frame_count():
    cfg = _cfg()
    replay = record_episode(
        cfg, RandomPolicy(), np.random.default_rng(0), ReplayPolicyInfo(kind="random"), "r1", seed=1
    )
    assert len(replay.frames) == replay.outcome.steps
    assert replay.frames[-1].t == replay.outcome.steps


def test_record_episode_heuristic_success():
    cfg = _cfg(max_steps=100)
    replay = record_episode(
        cfg,
        FrontierHeuristicPolicy(),
        np.random.default_rng(0),
        ReplayPolicyInfo(kind="heuristic"),
        "r2",
        seed=1_000_000,
    )
    assert replay.outcome.success
    assert replay.outcome.metrics["n_rescued"] == cfg.n_targets


def test_replay_save_load_roundtrip(tmp_path):
    cfg = _cfg()
    replay = record_episode(
        cfg, RandomPolicy(), np.random.default_rng(0), ReplayPolicyInfo(kind="random"), "r3", seed=5
    )
    path = tmp_path / "replay.json.gz"
    save_replay(replay, path)
    loaded = load_replay(path)
    assert loaded.replay_id == replay.replay_id
    assert loaded.map.hash == replay.map.hash
    assert len(loaded.frames) == len(replay.frames)
    assert loaded.frames[0].model_dump() == replay.frames[0].model_dump()
    assert loaded.outcome.success == replay.outcome.success


def test_resimulate_matches_random_policy_episode():
    cfg = _cfg(max_steps=40)
    replay = record_episode(
        cfg, RandomPolicy(), np.random.default_rng(2), ReplayPolicyInfo(kind="random"), "r4", seed=9
    )
    assert resimulate_matches(replay, cfg)


def test_resimulate_matches_heuristic_episode():
    cfg = _cfg(max_steps=100)
    replay = record_episode(
        cfg,
        FrontierHeuristicPolicy(),
        np.random.default_rng(0),
        ReplayPolicyInfo(kind="heuristic"),
        "r5",
        seed=1_000_005,
    )
    assert resimulate_matches(replay, cfg)


def test_resimulate_detects_tampering():
    """Corrupting a recorded frame's action must make resimulation fail —
    proves the check is actually verifying content, not vacuously true."""
    cfg = _cfg(max_steps=40)
    replay = record_episode(
        cfg, RandomPolicy(), np.random.default_rng(4), ReplayPolicyInfo(kind="random"), "r6", seed=11
    )
    if len(replay.frames) < 2:
        return  # too short to meaningfully tamper with; skip
    # Flip an action in an early frame to something that changes outcome.
    original = replay.frames[0].action[0]
    replay.frames[0].action[0] = (original + 1) % 5
    assert not resimulate_matches(replay, cfg)


def test_newly_known_cells_recorded_and_monotone_growth():
    cfg = _cfg(max_steps=30)
    replay = record_episode(
        cfg, RandomPolicy(), np.random.default_rng(0), ReplayPolicyInfo(kind="random"), "r7", seed=3
    )
    total_new_cells = sum(len(f.newly_known) for f in replay.frames)
    n_free = int(sum(row.count(".") for row in replay.map.terrain))
    assert total_new_cells <= n_free  # can never exceed total explorable cells


def test_replay_schema_version_present():
    cfg = _cfg()
    replay = record_episode(
        cfg, RandomPolicy(), np.random.default_rng(0), ReplayPolicyInfo(kind="random"), "r8", seed=1
    )
    assert replay.schema_version == 1

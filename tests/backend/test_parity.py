"""Backend <-> evaluator parity (§28.4, Phase 10 gate).

A live session with (map, seed, stochastic) must produce the identical
action/frame sequence as `marl.evaluation` run with the same inputs. The
real-artifact test goes further: it reproduces a stored *study* replay."""

from __future__ import annotations

import numpy as np
import pytest

from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
from marl.envs.core.config import LevelRegistry
from marl.evaluation.evaluator import run_episode
from marl.evaluation.final_eval import episode_seed, evaluate_policy
from marl.replay.loader import load_replay
from marl.replay.recorder import record_episode
from marl.replay.schema import ReplayPolicyInfo
from tests.backend.conftest import REAL_REPLAY_ID, RUN_ID


def _drive(client, body):
    sid = client.post("/api/sessions", json=body).json()["session_id"]
    frames = []
    while True:
        r = client.post(f"/api/sessions/{sid}/step", json={"n": 50}).json()
        frames += r["frames"]
        if r["done"]:
            return frames, r


@pytest.mark.parametrize("stochastic", [True, False])
def test_learned_session_equals_record_episode(client, results_tree, stochastic):
    """Fixed validation map, tiny exported actor: frames must be identical."""
    cfg = LevelRegistry().get("L1")
    m = client.get("/api/maps", params={"set": "validation", "level": "L1"}).json()[2]
    body = {"level": "L1", "map": {"kind": "fixed", "set": "validation", "id": m["id"]},
            "policy": {"kind": "ippo", "model_id": RUN_ID}, "stochastic": stochastic, "seed": 7}
    frames, last = _drive(client, body)

    actor, _ = load_actor(str(results_tree / "results" / "runs" / "demo" / "ippo" / "L1" / RUN_ID / "actor.pt"))
    from marl.envs.core.maps_io import MapSpec

    spec = MapSpec(m["id"], m["width"], m["height"], m["terrain"], [tuple(p) for p in m["robot_starts"]], [tuple(p) for p in m["targets"]])
    ref = record_episode(
        cfg, LearnedActorPolicy(actor, stochastic=stochastic), np.random.default_rng(episode_seed(m["id"], 7)),
        ReplayPolicyInfo(kind="ippo"), "ref", map_spec=spec,
    )
    assert [f.model_dump() for f in ref.frames] == frames  # every field of every frame
    assert last["final_metrics"] == ref.outcome.metrics
    assert last["step_index"] == ref.outcome.steps and last["metrics"]["success"] == ref.outcome.success


def test_session_equals_run_episode_trace_and_batch_aggregate(client):
    """Same inputs through the batch evaluator (`evaluate_policy`) -> same episode record."""
    from marl.algorithms.baselines.random_policy import RandomPolicy
    from marl.envs.core.maps_io import MapSpec

    cfg = LevelRegistry().get("L1")
    m = client.get("/api/maps", params={"set": "validation", "level": "L1"}).json()[0]
    spec = MapSpec(m["id"], m["width"], m["height"], m["terrain"], [tuple(p) for p in m["robot_starts"]], [tuple(p) for p in m["targets"]])
    frames, last = _drive(client, {"level": "L1", "map": {"kind": "fixed", "set": "validation", "id": m["id"]},
                                   "policy": {"kind": "random"}, "seed": 4})
    trace = run_episode(cfg, RandomPolicy(), np.random.default_rng(episode_seed(m["id"], 4)), map_spec=spec).trace
    assert [f["action"] for f in frames] == [a.tolist() for a in trace.actions]
    assert [f["pos"] for f in frames] == [p.tolist() for p in trace.positions[1:]]
    batch = evaluate_policy(cfg, RandomPolicy, [spec], eval_seeds=[4])["episodes"][0]
    assert last["metrics"]["success"] == batch["success"] and last["step_index"] == (batch["steps"] if batch["success"] else cfg.max_steps)
    assert last["final_metrics"]["n_collisions"] == batch["collisions"]


def test_procedural_map_session_equals_run_episode_by_seed(client):
    from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy

    cfg = LevelRegistry().get("L1")
    snap = client.post("/api/sessions", json={"level": "L1", "policy": {"kind": "heuristic"}, "seed": 11}).json()
    frames, _ = _drive(client, {"level": "L1", "policy": {"kind": "heuristic"}, "seed": 11})
    trace = run_episode(cfg, FrontierHeuristicPolicy(), np.random.default_rng(episode_seed(snap["map"]["id"], 11)), seed=11).trace
    assert [f["action"] for f in frames] == [a.tolist() for a in trace.actions]


def test_real_mappo_session_reproduces_stored_study_replay(real_client):
    """The strongest check: a live session driven by the real trained MAPPO actor
    reproduces the replay the study recorded (Phase 7), frame for frame."""
    entry = next((r for r in real_client.get("/api/replays").json() if r["replay_id"] == REAL_REPLAY_ID), None)
    if entry is None:
        pytest.skip("study replay not present")
    stored = load_replay(
        __import__("backend.app.settings", fromlist=["settings"]).settings.replays_dir / entry["study"] / entry["run_id"] / f"{REAL_REPLAY_ID}.json.gz"
    )
    body = {"level": "L2", "map": {"kind": "fixed", "set": "structured", "id": stored.map.id},
            "policy": {"kind": "mappo", "model_id": entry["run_id"]}, "stochastic": True, "seed": 0}
    frames, last = _drive(real_client, body)
    assert frames == [f.model_dump() for f in stored.frames]
    # The known map a viewer rebuilds from the stored replay (derived `initial` + deltas)
    # equals the live session's own final known map: the Phase 11 reconstruction rule is sound.
    init = real_client.get(f"/api/replays/{REAL_REPLAY_ID}/initial").json()
    assert init["source"] == "derived"  # study replays predate the field
    known = {tuple(c) for c in init["known_team"]} | {tuple(c) for f in stored.frames for c in f.newly_known}
    sid = real_client.post("/api/sessions", json=body).json()["session_id"]
    done = False
    while not done:
        done = real_client.post(f"/api/sessions/{sid}/step", json={"n": 50}).json()["done"]
    assert known == {tuple(c) for c in real_client.get(f"/api/sessions/{sid}").json()["current_known_team"]}
    assert last["final_metrics"] == stored.outcome.metrics
    assert last["step_index"] == stored.outcome.steps and last["metrics"]["success"] == stored.outcome.success

"""Live session REST API (§28.2) and its resource limits (§37)."""

from __future__ import annotations

import time

from marl.replay.schema import ReplayFrame
from tests.backend.conftest import RUN_ID

BASE = {"level": "L1", "policy": {"kind": "heuristic"}, "seed": 3}


def _create(client, **over):
    return client.post("/api/sessions", json={**BASE, **over})


def _run_to_end(client, sid):
    frames = []
    while True:
        r = client.post(f"/api/sessions/{sid}/step", json={"n": 50}).json()
        frames += r["frames"]
        if r["done"]:
            return frames, r


def test_create_returns_initial_snapshot(client):
    r = _create(client)
    assert r.status_code == 201
    s = r.json()
    assert s["step_index"] == 0 and s["done"] is False and s["last_frame"] is None
    assert s["map"]["width"] == 8 and len(s["map"]["terrain"]) == 8
    assert s["initial"]["pos"] == s["current_pos"] == s["map"]["robot_starts"]
    assert s["initial"]["known_team"], "initial sensing pass must be exposed (it is not in any frame delta)"
    assert s["policy"]["kind"] == "heuristic" and s["policy"]["eval_seed"] == 3
    assert s["metrics"]["t"] == 0 and s["target_status"] == [0]


def test_step_returns_replay_frames_and_get_matches(client):
    sid = _create(client).json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/step", json={"n": 3}).json()
    assert [f["t"] for f in r["frames"]] == [1, 2, 3] and r["step_index"] == 3
    for f in r["frames"]:
        ReplayFrame.model_validate(f)  # one frame type for live and recorded
    snap = client.get(f"/api/sessions/{sid}").json()
    assert snap["step_index"] == 3 and snap["last_frame"] == r["frames"][-1]
    assert len(snap["current_known_team"]) >= len(snap["initial"]["known_team"])


def test_step_default_body_and_bounds(client):
    sid = _create(client).json()["session_id"]
    assert len(client.post(f"/api/sessions/{sid}/step").json()["frames"]) == 1
    assert client.post(f"/api/sessions/{sid}/step", json={"n": 0}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/step", json={"n": 51}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/step", json={"n": "x"}).status_code == 422


def test_episode_end_final_metrics_then_409_and_reset(client):
    sid = _create(client).json()["session_id"]
    frames, last = _run_to_end(client, sid)
    assert last["done"] and last["final_metrics"]["n_collisions"] >= 0
    assert last["metrics"]["success"] is True and len(frames) == last["step_index"]
    assert client.post(f"/api/sessions/{sid}/step").status_code == 409
    snap = client.post(f"/api/sessions/{sid}/reset").json()  # same (map, seed): same episode
    assert snap["step_index"] == 0 and snap["done"] is False and snap["final_metrics"] is None
    frames2, _ = _run_to_end(client, sid)
    assert frames2 == frames  # deterministic restart


def test_reset_with_new_seed_changes_procedural_map(client):
    s1 = _create(client).json()
    s2 = client.post(f"/api/sessions/{s1['session_id']}/reset", json={"seed": 99}).json()
    assert s2["seed"] == 99 and s2["map"]["id"] != s1["map"]["id"]
    assert s2["policy"]["eval_seed"] == 99


def test_fixed_validation_map(client):
    maps = client.get("/api/maps", params={"set": "validation", "level": "L1"}).json()
    r = _create(client, map={"kind": "fixed", "set": "validation", "id": maps[0]["id"]})
    assert r.status_code == 201 and r.json()["map"]["id"] == maps[0]["id"]
    assert r.json()["map"]["terrain"] == maps[0]["terrain"]
    assert _create(client, map={"kind": "fixed", "set": "validation", "id": "no-such-map"}).status_code == 404
    assert _create(client, map={"kind": "fixed", "set": "test", "id": "x/../y"}).status_code == 422


def test_learned_policy_session(client):
    r = _create(client, policy={"kind": "ippo", "model_id": RUN_ID}, stochastic=False)
    assert r.status_code == 201
    pol = r.json()["policy"]
    assert pol["kind"] == "ippo" and pol["run_id"] == RUN_ID and pol["checkpoint_step"] == 2048 and pol["stochastic"] is False
    frames = client.post(f"/api/sessions/{r.json()['session_id']}/step", json={"n": 5}).json()["frames"]
    assert len(frames) == 5


def test_policy_validation_errors(client):
    assert _create(client, policy={"kind": "mappo"}).status_code == 422  # model_id required
    assert _create(client, policy={"kind": "random", "model_id": RUN_ID}).status_code == 422
    assert _create(client, policy={"kind": "bogus"}).status_code == 422
    assert _create(client, policy={"kind": "ippo", "model_id": "missing"}).status_code == 404
    assert _create(client, policy={"kind": "mappo", "model_id": RUN_ID}).status_code == 422  # it is an ippo model
    assert _create(client, level="L2", policy={"kind": "ippo", "model_id": RUN_ID}).status_code == 422  # trained on L1


def test_level_xor_env_config(client):
    assert client.post("/api/sessions", json={"policy": {"kind": "random"}}).status_code == 422
    cfg = client.get("/api/levels").json()[0]["config"]
    both = client.post("/api/sessions", json={"level": "L1", "env_config": cfg, "policy": {"kind": "random"}})
    assert both.status_code == 422
    ok = client.post("/api/sessions", json={"env_config": {**cfg, "name": "custom", "max_steps": 20}, "policy": {"kind": "random"}})
    assert ok.status_code == 201 and ok.json()["env"]["level"] == "custom"
    bad = client.post("/api/sessions", json={"env_config": {**cfg, "width": 500}, "policy": {"kind": "random"}})
    assert bad.status_code == 422  # EnvConfig bounds apply
    assert client.post("/api/sessions", json={"level": "L9", "policy": {"kind": "random"}}).status_code == 404


def test_unknown_and_malformed_session_ids(client):
    assert client.get("/api/sessions/deadbeef").status_code == 404
    assert client.post("/api/sessions/deadbeef/step").status_code == 404
    assert client.delete("/api/sessions/deadbeef").status_code == 404
    assert client.get("/api/sessions/.hidden").status_code == 422


def test_delete_closes_session(client):
    sid = _create(client).json()["session_id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.get("/api/health").json()["counts"]["live_sessions"] == 0


def test_max_live_sessions_enforced(make_client):
    c = make_client(max_live_sessions=2)
    ids = [_create(c).json()["session_id"] for _ in range(2)]
    assert _create(c).status_code == 429
    c.delete(f"/api/sessions/{ids[0]}")
    assert _create(c).status_code == 201


def test_idle_ttl_expires_sessions(make_client):
    c = make_client(session_idle_ttl_seconds=0)
    sid = _create(c).json()["session_id"]
    time.sleep(0.05)
    assert c.get(f"/api/sessions/{sid}").status_code == 404

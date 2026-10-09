"""End-to-end stack tests over a real server + the committed demo bundle
(Phase 14). Every assertion is against real artifacts: trained actors,
stored study replays, study summaries."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import websockets

from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
from marl.envs.core.config import LevelRegistry
from marl.replay.loader import load_replay
from marl.utils.paths import REPO_ROOT
from tests.integration.conftest import ORIGIN, ws_url

pytestmark = pytest.mark.integration
STUDY = "main_L2"


def test_bundle_is_self_consistent_and_tamper_evident(tmp_path: Path):
    res = tmp_path / "results"
    setup = [sys.executable, str(REPO_ROOT / "scripts" / "demo_setup.py"), "--results", str(res)]
    assert subprocess.run(setup, capture_output=True).returncode == 0
    assert subprocess.run([*setup, "--check"], capture_output=True).returncode == 0
    victim = next(res.rglob("actor.pt"))
    raw = bytearray(victim.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # flip one byte
    victim.write_bytes(bytes(raw))
    bad = subprocess.run([*setup, "--check"], capture_output=True, text=True)
    assert bad.returncode == 1 and "differs from bundle" in bad.stderr
    # install never overwrites a differing file silently
    assert subprocess.run(setup, capture_output=True).returncode == 1
    shutil.rmtree(res)


def test_catalogue_comes_from_the_bundle_only(api):
    h = api.get("/health").json()
    assert h["counts"]["runs"] == 10 and h["counts"]["models"] == 10 and h["counts"]["replays"] == 40
    models = api.get("/models").json()
    assert sorted(m["algo"] for m in models) == ["ippo"] * 5 + ["mappo"] * 5
    runs = api.get("/experiments").json()
    assert len(runs) == 10 and {r["status"] for r in runs} == {"completed"} and {r["study"] for r in runs} == {STUDY}
    assert [s["study"] for s in api.get("/studies").json()] == [STUDY]


def test_no_endpoint_leaks_a_local_path(api):
    """Stored manifests contain absolute paths of the machine that trained them."""
    rid = api.get("/experiments").json()[0]["run_id"]
    paths = ["/health", "/models", f"/models/{rid}", "/experiments", f"/experiments/{rid}", f"/experiments/{rid}/eval",
             "/studies", f"/studies/{STUDY}", "/replays", f"/studies/{STUDY}/analysis/L2_coordination", "/preregistration"]
    leaked = [p for p in paths if re.search(r"/Users/|/home/|C:\\\\", api.get(p).text)]
    assert leaked == []


def _numbers(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _numbers(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _numbers(v, f"{prefix}/{i}")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, x


def test_every_api_number_is_the_number_in_the_results_file(api, bundle_results: Path):
    """Traceability: the study summary and a run's final test metrics served by the API
    are, number for number, the files the pipeline wrote."""
    on_disk = json.loads((bundle_results / "studies" / STUDY / "summary.json").read_text())
    served = api.get(f"/studies/{STUDY}").json()["summary"]
    a, b = dict(_numbers(on_disk)), dict(_numbers(served))
    assert a and a == b
    run = next(r for r in api.get("/experiments").json() if r["algo"] == "mappo")
    ev = json.loads((bundle_results / "runs" / STUDY / "mappo" / "L2" / run["run_id"] / "eval" / "test_L2.json").read_text())
    model = next(m for m in api.get("/models").json() if m["model_id"] == run["run_id"])
    stoch = ev["modes"]["stochastic"]["aggregate"] if "modes" in ev else ev["stochastic"]["aggregate"]
    assert model["test_metrics"]["success_rate"] == stoch["success_rate"]


def _final_replays(api):
    return [r for r in api.get("/replays").json() if r["fraction"] == 1.0]


def _drive_ws(server: str, session_id: str) -> list[dict]:
    async def run():
        frames = []
        async with websockets.connect(ws_url(server, f"/ws/sessions/{session_id}"), origin=ORIGIN) as ws:
            assert json.loads(await ws.recv())["type"] == "state"
            await ws.send(json.dumps({"cmd": "speed", "value": 8}))
            assert json.loads(await ws.recv())["type"] == "state"
            await ws.send(json.dumps({"cmd": "play"}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if msg["type"] == "frame":
                    frames.append(msg["frame"])
                elif msg["type"] == "episode_end":
                    return frames
                elif msg["type"] == "error":
                    raise AssertionError(msg)

    return asyncio.run(run())


def test_live_ws_sessions_reproduce_every_stored_final_replay(api, server, bundle_results: Path):
    """model -> env -> backend -> socket: for each of the 10 stored final-checkpoint replays
    (IPPO and MAPPO), a live WebSocket session with the same map, model and seed streams
    exactly the frames the study recorded."""
    replays = _final_replays(api)
    assert len(replays) == 10
    for entry in replays:
        stored = load_replay(bundle_results / "replays" / entry["study"] / entry["run_id"] / f"{entry['replay_id']}.json.gz")
        body = {"level": "L2", "map": {"kind": "fixed", "set": "structured", "id": stored.map.id},
                "policy": {"kind": entry["algo"], "model_id": entry["run_id"]}, "stochastic": True, "seed": entry["seed"]}
        r = api.post("/sessions", json=body)
        assert r.status_code in (200, 201), r.text
        frames = _drive_ws(server, r.json()["session_id"])
        assert frames == [f.model_dump() for f in stored.frames], entry["replay_id"]
        api.delete(f"/sessions/{r.json()['session_id']}")


def test_live_session_equals_direct_evaluator_run(api, server, bundle_results: Path):
    """Same check against the evaluator directly (no stored replay): a fresh procedural map."""
    from marl.evaluation.final_eval import episode_seed
    from marl.replay.recorder import record_episode
    from marl.replay.schema import ReplayPolicyInfo

    model = next(m for m in api.get("/models").json() if m["algo"] == "mappo")
    body = {"level": "L2", "policy": {"kind": "mappo", "model_id": model["model_id"]}, "stochastic": False, "seed": 123}
    snap = api.post("/sessions", json=body).json()
    frames = _drive_ws(server, snap["session_id"])
    actor, _ = load_actor(str(bundle_results / "runs" / STUDY / "mappo" / "L2" / model["model_id"] / "actor.pt"))
    from marl.envs.core.maps_io import MapSpec

    m = snap["map"]
    spec = MapSpec(m["id"], m["width"], m["height"], m["terrain"], [tuple(p) for p in m["robot_starts"]], [tuple(p) for p in m["targets"]])
    ref = record_episode(LevelRegistry().get("L2"), LearnedActorPolicy(actor, stochastic=False),
                         np.random.default_rng(episode_seed(m["id"], 123)), ReplayPolicyInfo(kind="mappo"), "ref", map_spec=spec)
    assert frames == [f.model_dump() for f in ref.frames]


def test_ws_security_over_real_sockets(api, server):
    sid = api.post("/sessions", json={"level": "L1", "policy": {"kind": "random"}, "seed": 1}).json()["session_id"]

    async def close_code(url, **kw):
        async with websockets.connect(url, **kw) as ws:
            try:
                await asyncio.wait_for(ws.recv(), 5)
                await asyncio.wait_for(ws.recv(), 5)
            except websockets.ConnectionClosed as e:
                return e.code
        return None

    assert asyncio.run(close_code(ws_url(server, f"/ws/sessions/{sid}"), origin="http://evil.example")) == 4403
    assert asyncio.run(close_code(ws_url(server, "/ws/sessions/does-not-exist"), origin=ORIGIN)) == 4404
    # Host-header allow-list (DNS rebinding)
    import httpx

    assert httpx.get(f"{server}/api/health", headers={"Host": "evil.example"}).status_code in (400, 403)
    # form posts are refused: JSON only
    assert api.post("/sessions", content="level=L1", headers={"content-type": "application/x-www-form-urlencoded"}).status_code == 422


def test_training_job_streams_live_metrics_then_cancels(api, server, bundle_results: Path):
    """UI flow: launch the (smoke-sized) trainer, watch live metrics on the training socket, cancel."""
    req = {"level": "L1", "algo": "ippo", "seed": 0, "total_env_steps": 400_000,
           "overrides": {"ppo.n_envs": 4, "ppo.rollout_len": 64, "ppo.epochs": 1}}
    r = api.post("/training/jobs", json=req)
    assert r.status_code == 202, r.text
    job = r.json()
    deadline = time.time() + 90
    while not job.get("run_id") and time.time() < deadline:  # the run dir appears once the trainer has started
        time.sleep(0.5)
        job = api.get(f"/training/jobs/{job['job_id']}").json()
    run_id = job["run_id"]
    assert run_id, job

    async def watch():
        got = []
        async with websockets.connect(ws_url(server, f"/ws/training/{run_id}"), origin=ORIGIN) as ws:
            deadline = time.time() + 90
            while len([m for m in got if m["type"] == "update"]) < 3 and time.time() < deadline:
                got.append(json.loads(await asyncio.wait_for(ws.recv(), 60)))
            c = api.post(f"/training/jobs/{job['job_id']}/cancel")
            assert c.status_code == 200, c.text
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), 60))
                got.append(m)
                if m["type"] == "end":
                    return got

    got = asyncio.run(watch())
    updates = [m for m in got if m["type"] == "update"]
    assert len(updates) >= 3 and [u["update"] for u in updates] == sorted(u["update"] for u in updates)
    assert got[-1] == {"type": "end", "status": "interrupted"}
    status = api.get(f"/training/jobs/{job['job_id']}").json()
    assert status["status"] == "cancelled"
    manifest = json.loads(next((bundle_results / "runs" / "ui").glob(f"*/*/{run_id}/manifest.json")).read_text())
    assert manifest["status"] == "interrupted"
    # The metrics the socket pushed are the rows the trainer wrote
    rows = [json.loads(line) for line in next((bundle_results / "runs" / "ui").glob(f"*/*/{run_id}/metrics.jsonl")).read_text().splitlines()]
    by_update = {r["update"]: r for r in rows}
    assert all(by_update[u["update"]]["mean_reward"] == u["mean_reward"] for u in updates)


def test_demo_setup_refuses_a_manifest_that_escapes_results(tmp_path: Path):
    bundle = tmp_path / "bundle"
    (bundle / "results").mkdir(parents=True)
    (bundle / "MANIFEST.json").write_text(json.dumps({"n_files": 1, "files": {"../../evil": {"sha256": "0" * 64, "bytes": 1}}}))
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "demo_setup.py"), "--bundle", str(bundle), "--results", str(tmp_path / "res")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "unsafe path" in r.stderr and not (tmp_path / "evil").exists()


@pytest.mark.skipif(not (REPO_ROOT / "frontend" / "dist" / "index.html").is_file(), reason="needs `make build-frontend` first")
def test_demo_launcher_serves_both_apps_and_stops_only_its_own_children():
    """Regression: the launcher used `kill 0`, which killed its caller's whole process group."""
    import httpx

    def up(url: str) -> bool:
        try:
            return httpx.get(url, timeout=1).status_code == 200
        except httpx.HTTPError:
            return False

    if up("http://127.0.0.1:8000/api/health") or up("http://localhost:5173/"):
        pytest.skip("ports 8000/5173 are in use")
    flag = Path(tempfile.mkdtemp()) / "stop"
    script = f"{REPO_ROOT}/scripts/demo.sh >/dev/null 2>&1 & pid=$!; for i in $(seq 120); do [ -e {flag} ] && break; sleep 0.5; done; kill $pid; wait $pid; echo CALLER-SURVIVED"
    parent = subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not (up("http://127.0.0.1:8000/api/health") and up("http://localhost:5173/")):
            time.sleep(0.25)
        assert up("http://127.0.0.1:8000/api/health") and up("http://localhost:5173/")
    finally:
        flag.touch()  # tell the caller script to stop the launcher now
        out, _ = parent.communicate(timeout=60)
    assert "CALLER-SURVIVED" in out
    time.sleep(1)
    assert not up("http://127.0.0.1:8000/api/health") and not up("http://localhost:5173/")

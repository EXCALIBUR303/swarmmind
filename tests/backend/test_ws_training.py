"""Training WebSocket: tails metrics.jsonl / eval_val.jsonl (§28.3)."""

from __future__ import annotations

import json

import pytest
from starlette.websockets import WebSocketDisconnect

from backend.app.api.ws.train_ws import JsonlTail
from tests.backend.conftest import RUN_ID, write_json

RUNNING = "s1-20260101T000000Z-bbbb2222"


def _recv_until(ws, kind: str, limit: int = 40):
    out = []
    for _ in range(limit):
        m = ws.receive_json()
        out.append(m)
        if m["type"] == kind:
            return out
    raise AssertionError([m["type"] for m in out])


def test_jsonl_tail_waits_for_complete_lines(tmp_path):
    f = tmp_path / "m.jsonl"
    tail = JsonlTail(f)
    assert tail.read_new() == []  # file missing
    f.write_text('{"update": 0}\n{"update": 1')
    assert tail.read_new() == [{"update": 0}]  # half-written line is held back
    with f.open("a") as fh:
        fh.write('}\nnot-json\n{"update": 2}\n')
    assert tail.read_new() == [{"update": 1}, {"update": 2}]  # corrupt line skipped
    assert tail.read_new() == []


def test_finished_run_replays_history_then_ends(client):
    with client.websocket_connect(f"/ws/training/{RUN_ID}") as ws:
        msgs = _recv_until(ws, "end")
    kinds = [m["type"] for m in msgs]
    assert kinds.count("update") == 3 and kinds.count("val_eval") == 1
    assert {"type": "status", "status": "completed"} in msgs
    assert msgs[-1] == {"type": "end", "status": "completed"}


def test_from_update_skips_old_updates(client):
    with client.websocket_connect(f"/ws/training/{RUN_ID}?from_update=2") as ws:
        msgs = _recv_until(ws, "end")
    assert [m["update"] for m in msgs if m["type"] == "update"] == [2]


def test_live_tail_pushes_new_lines_and_status_changes(client, results_tree):
    run = results_tree / "results" / "runs" / "demo" / "mappo" / "L1" / RUNNING
    with client.websocket_connect(f"/ws/training/{RUNNING}") as ws:
        first = _recv_until(ws, "status")
        assert first[-1]["status"] == "running"
        with (run / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps({"update": 3, "env_steps": 1024}) + "\n")
            fh.write('{"update": 4, "env_')  # partial
            fh.flush()
        got = [m for m in _recv_until(ws, "update") if m["type"] == "update"]
        assert got[-1]["update"] == 3
        with (run / "metrics.jsonl").open("a") as fh:
            fh.write('steps": 1280}\n')
        assert _recv_until(ws, "update")[-1]["update"] == 4
        manifest = json.loads((run / "manifest.json").read_text())
        write_json(run / "manifest.json", {**manifest, "status": "completed"})
        tail = _recv_until(ws, "end")
        assert {"type": "status", "status": "completed"} in tail


def test_unknown_or_unsafe_run_closes_4404(client):
    for rid in ("missing", ".hidden"):
        with pytest.raises(WebSocketDisconnect) as e, client.websocket_connect(f"/ws/training/{rid}"):
            pass
        assert e.value.code == 4404


def test_client_disconnect_is_clean(client):
    with client.websocket_connect(f"/ws/training/{RUNNING}") as ws:
        _recv_until(ws, "status")
    assert client.get("/api/health").status_code == 200

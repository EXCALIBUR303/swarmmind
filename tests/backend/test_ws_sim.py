"""Simulation WebSocket protocol (§28.3)."""

from __future__ import annotations

import json
import time

import pytest
from starlette.websockets import WebSocketDisconnect

from backend.app.api.ws.sim_ws import _Player
from tests.backend.conftest import ORIGIN


def _session(client, **over):
    body = {"level": "L1", "policy": {"kind": "heuristic"}, "seed": 3, **over}
    return client.post("/api/sessions", json=body).json()["session_id"]


def _until(ws, wanted: str, limit: int = 400):
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m)
        if m["type"] == wanted:
            return seen
    raise AssertionError(f"never saw {wanted}: {[m['type'] for m in seen][-5:]}")


def test_connect_sends_resync_state(client):
    sid = _session(client)
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        m = ws.receive_json()
    assert m["type"] == "state" and m["playing"] is False and m["speed"] == 1.0
    assert m["snapshot"]["session_id"] == sid and m["snapshot"]["step_index"] == 0


def test_step_command_streams_frame_messages(client):
    sid = _session(client)
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "step", "value": 2})
        a, b = ws.receive_json(), ws.receive_json()
    assert (a["type"], a["frame"]["t"], a["step_index"]) == ("frame", 1, 1)
    assert b["frame"]["t"] == 2 and b["metrics"]["t"] == 2
    assert client.get(f"/api/sessions/{sid}").json()["step_index"] == 2  # same server-side session


def test_play_runs_to_episode_end_then_play_is_an_error(client):
    sid = _session(client)
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "speed", "value": 8})
        assert ws.receive_json()["speed"] == 8.0
        ws.send_json({"cmd": "play"})
        msgs = _until(ws, "episode_end")
        frames = [m for m in msgs if m["type"] == "frame"]
        assert [f["frame"]["t"] for f in frames] == list(range(1, len(frames) + 1))
        end = msgs[-1]
        assert end["success"] is True and end["steps"] == len(frames) and "n_collisions" in end["metrics"]
        ws.send_json({"cmd": "play"})
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "episode_done"
        ws.send_json({"cmd": "step"})
        assert ws.receive_json()["code"] == "episode_done"


def test_pause_stops_the_stream_and_resumes_without_gaps(client):
    sid = _session(client, seed=5, policy={"kind": "random"})  # long episode
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "play"})
        ws.receive_json()  # state(playing)
        first = _until(ws, "frame")[-1]
        ws.send_json({"cmd": "pause"})
        msgs = _until(ws, "state")
        assert msgs[-1]["playing"] is False
        last_t = ([first] + [m for m in msgs if m["type"] == "frame"])[-1]["frame"]["t"]
        assert msgs[-1]["snapshot"]["step_index"] == last_t  # nothing advanced unseen by the client
        ws.send_json({"cmd": "step"})
        assert ws.receive_json()["frame"]["t"] == last_t + 1


def test_reset_over_ws_resyncs_state(client):
    sid = _session(client)
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "step", "value": 3})
        _until(ws, "frame")
        ws.receive_json()
        ws.receive_json()
        ws.send_json({"cmd": "reset", "value": {"seed": 8}})
        m = _until(ws, "state")[-1]
    assert m["snapshot"]["step_index"] == 0 and m["snapshot"]["seed"] == 8


def test_bad_commands_are_errors_not_disconnects(client):
    sid = _session(client)
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ws.receive_json()
        for payload, code in [
            ({"cmd": "explode"}, "bad_command"),
            ({"nope": 1}, "bad_command"),
            ({"cmd": "speed", "value": 99}, "bad_value"),
            ({"cmd": "speed", "value": "fast"}, "bad_command"),
            ({"cmd": "speed", "value": True}, "bad_command"),
            ({"cmd": "step", "value": 51}, "bad_value"),
            ({"cmd": "step", "value": 0}, "bad_value"),
            ({"cmd": "step", "value": 1.5}, "bad_value"),
            ({"cmd": "reset", "value": {"seed": -1}}, "bad_reset"),
        ]:
            ws.send_json(payload)
            m = ws.receive_json()
            assert (m["type"], m.get("code")) == ("error", code), (payload, m)
        ws.send_text("{not json")
        assert ws.receive_json()["code"] == "bad_command"
        ws.send_text(" " * 5000)
        assert ws.receive_json()["code"] == "message_too_large"
        ws.send_json({"cmd": "step"})  # connection is still healthy
        assert ws.receive_json()["type"] == "frame"


def test_step_while_playing_is_refused(client):
    sid = _session(client, seed=5, policy={"kind": "random"})
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "play"})
        ws.send_json({"cmd": "step"})
        assert any(m.get("code") == "playing" for m in _until(ws, "error"))
        ws.send_json({"cmd": "pause"})


def test_unknown_or_malformed_session_closes_4404(client):
    for sid in ("deadbeef", ".hidden"):
        with client.websocket_connect(f"/ws/sessions/{sid}") as ws, pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 4404


def test_server_owns_the_clock_and_caps_rate(client):
    # rate(): 4 steps/s at 1x, capped at 30/s even at 8x (4*8=32)
    class _P(_Player):
        def __init__(self): self.max_sps, self.base_sps, self.speed = 30, 4.0, 1.0
    p = _P()
    assert p.rate() == 4.0
    p.speed = 8.0
    assert p.rate() == 30.0
    p.speed = 0.25
    assert p.rate() == 1.0

    sid = _session(client, seed=5, policy={"kind": "random"})
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "speed", "value": 8})
        ws.receive_json()
        ws.send_json({"cmd": "play"})
        ws.receive_json()
        t0 = time.monotonic()
        for _ in range(31):
            _until(ws, "frame")
        elapsed = time.monotonic() - t0
        ws.send_json({"cmd": "pause"})
    assert elapsed >= 30 / 30 * 0.9, f"31 frames in {elapsed:.2f}s exceeds the 30 steps/s cap"


def test_rest_and_ws_share_one_session_state(client):
    sid = _session(client)
    client.post(f"/api/sessions/{sid}/step", json={"n": 2})
    with client.websocket_connect(f"/ws/sessions/{sid}", headers={"origin": ORIGIN}) as ws:
        assert ws.receive_json()["snapshot"]["step_index"] == 2


def test_new_connection_takes_over_and_old_one_is_closed(client):
    sid = _session(client)
    with client.websocket_connect(f"/ws/sessions/{sid}") as old:
        old.receive_json()
        with client.websocket_connect(f"/ws/sessions/{sid}") as new:
            assert new.receive_json()["type"] == "state"
            with pytest.raises(WebSocketDisconnect) as e:
                old.receive_json()
            assert e.value.code == 4409
            new.send_json({"cmd": "step"})
            assert new.receive_json()["type"] == "frame"


def test_policy_fault_during_play_is_a_generic_error_not_a_crash(client):
    sid = _session(client)
    session = client.app.state.services.sessions.get(sid)

    class Boom:
        def act(self, world, rng):
            raise RuntimeError("secret internal detail /Users/x")

    session.policy = Boom()
    with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
        ws.receive_json()
        ws.send_json({"cmd": "play"})
        ws.receive_json()
        err = ws.receive_json()
        assert (err["type"], err["code"]) == ("error", "internal_error")
        assert "secret" not in json.dumps(err) and "/Users" not in json.dumps(err)
        ws.send_json({"cmd": "pause"})
        assert ws.receive_json()["type"] == "state"

"""Security properties of the local-first API (§37)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.main import create_app
from backend.app.settings import Settings
from marl.utils.paths import UnsafePathError, is_safe_id, safe_join
from tests.backend.conftest import ORIGIN, RUN_ID

TRAVERSAL = ["..", "%2e%2e", "..%2f..%2fetc%2fpasswd", "%2e%2e%2f%2e%2e%2fetc%2fpasswd", ".hidden", "a%00b", "a%5cb", "x" * 129]


@pytest.mark.parametrize(
    "path",
    ["/api/replays/{}", "/api/experiments/{}", "/api/experiments/{}/metrics", "/api/experiments/{}/eval",
     "/api/models/{}", "/api/studies/{}", "/api/sessions/{}", "/api/training/jobs/{}"],
)
@pytest.mark.parametrize("evil", TRAVERSAL)
def test_path_ids_reject_traversal(client, path, evil):
    r = client.get(path.format(evil))
    assert r.status_code in (404, 422), (path, evil, r.status_code)
    assert "root:" not in r.text


def test_query_ids_reject_traversal(client):
    for bad in ("../x", "a/b", ".."):
        assert client.get("/api/replays", params={"run_id": bad}).status_code == 422
        assert client.get("/api/maps", params={"set": "validation", "level": bad}).status_code == 422
        assert client.get("/api/analysis/demo/heatmap", params={"map_id": bad, "algo": "ippo"}).status_code == 422


def test_is_safe_id_rejects_dot_names():
    assert not is_safe_id(".") and not is_safe_id("..") and not is_safe_id(".hidden")
    assert is_safe_id("a.b-c_d") and is_safe_id("s0-20260924T090709Z-33bf6a2b")
    with pytest.raises(UnsafePathError):
        safe_join(__import__("pathlib").Path("/tmp"), "..")


def test_symlink_inside_replays_cannot_escape(client, results_tree, tmp_path):
    outside = tmp_path / "outside.json.gz"
    outside.write_bytes(b"secret")
    link = results_tree / "results" / "replays" / "demo" / RUN_ID / "evil.json.gz"
    link.symlink_to(outside)
    idx = link.parent / "index.json"
    import json

    entries = json.loads(idx.read_text())
    entries.append({"replay_id": "evil", "map_id": "m", "algo": "ippo"})
    idx.write_text(json.dumps(entries))
    assert client.get("/api/replays/evil").status_code == 404


def test_host_header_allowlist(results_tree):
    s = Settings(results_dir=results_tree / "results", models_dir=results_tree / "models", _env_file=None)
    with TestClient(create_app(s), base_url="http://evil.example") as c:
        assert c.get("/api/health").status_code == 400  # DNS-rebinding defence
    with TestClient(create_app(s), base_url="http://localhost") as c:
        assert c.get("/api/health").status_code == 200


def test_cors_only_for_configured_origin(client):
    ok = client.get("/api/health", headers={"Origin": ORIGIN})
    assert ok.headers["access-control-allow-origin"] == ORIGIN
    other = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers
    pre = client.options("/api/sessions", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "PUT"})
    assert pre.status_code == 400  # PUT is not an allowed method


def test_security_headers(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"


def test_ws_rejects_foreign_origin(client):
    sid = client.post("/api/sessions", json={"level": "L1", "policy": {"kind": "random"}}).json()["session_id"]
    # accepted only to be closed with a code a browser can see; no session data is ever sent
    with client.websocket_connect(f"/ws/sessions/{sid}", headers={"origin": "http://evil.example"}) as ws, pytest.raises(WebSocketDisconnect) as e:
        ws.receive_json()
    assert e.value.code == 4403
    with client.websocket_connect(f"/ws/sessions/{sid}", headers={"origin": ORIGIN}) as ws:
        assert ws.receive_json()["type"] == "state"


def test_errors_do_not_leak_filesystem_paths(client, results_tree):
    bodies = [
        client.get(f"/api/experiments/{RUN_ID}").text,
        client.get("/api/models/missing").text,
        client.get("/api/replays/missing").text,
        client.post("/api/sessions", json={"level": "L1", "policy": {"kind": "ippo", "model_id": "missing"}}).text,
    ]
    for b in bodies:
        assert str(results_tree) not in b and "/Users/" not in b


def test_corrupt_actor_is_a_clean_error(client, results_tree):
    actor = results_tree / "results" / "runs" / "demo" / "ippo" / "L1" / RUN_ID / "actor.pt"
    actor.write_bytes(b"not a torch file")
    r = client.post("/api/sessions", json={"level": "L1", "policy": {"kind": "ippo", "model_id": RUN_ID}})
    assert r.status_code == 422 and "could not be loaded" in r.json()["detail"]


def test_local_absolute_paths_are_scrubbed_from_responses(client):
    """Real manifests/summaries contain `run_dir: /Users/<name>/...`; none may reach a client."""
    import json
    from pathlib import Path

    from marl.utils.paths import REPO_ROOT

    for url in (f"/api/experiments/{RUN_ID}", f"/api/models/{RUN_ID}", f"/api/experiments/{RUN_ID}/eval", "/api/studies/demo"):
        body = client.get(url).text
        assert str(REPO_ROOT) not in body and str(Path.home()) not in body, url
    m = client.get(f"/api/experiments/{RUN_ID}").json()["manifest"]
    assert m["run_dir"].startswith("results/runs/demo/ippo/L1/") and m["home_note"].startswith("~/")
    assert json.dumps(client.get("/api/health").json()).count(str(Path.home())) == 0


def test_post_requires_json_content_type(client):
    """A cross-site 'simple' request (text/plain) cannot reach a JSON endpoint."""
    r = client.post("/api/sessions", content='{"level":"L1","policy":{"kind":"random"}}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 422
    assert client.get("/api/health").json()["counts"]["live_sessions"] == 0


def test_custom_env_config_is_bounded(client):
    cfg = client.get("/api/levels").json()[0]["config"]
    for patch in ({"max_steps": 10**9}, {"obs_radius": 100_000}, {"sensor_radius": 50.0, "obs_radius": 60}):
        r = client.post("/api/sessions", json={"env_config": {**cfg, "name": "x", **patch}, "policy": {"kind": "random"}})
        assert r.status_code == 422, patch
    assert client.get("/api/health").json()["counts"]["live_sessions"] == 0

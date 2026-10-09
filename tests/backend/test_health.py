"""Backend health endpoint test (§28.2)."""

from __future__ import annotations


def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "versions" in body
    assert body["device"] in ("cpu", "cuda", "unknown")
    counts = body["counts"]
    assert counts["runs"] == 3  # completed, running, and the corrupt one (listed, not hidden)
    assert counts["models"] == 1
    assert counts["replays"] == 1
    assert counts["live_sessions"] == 0

"""Backend health endpoint test (§28.2)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_ok():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "versions" in body
    assert "device" in body
    assert "counts" in body
    assert body["device"] in ("cpu", "cuda", "unknown")

"""The committed API contract (docs/openapi.json, schemas, API.md) must match the app."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("export_schemas", REPO / "scripts" / "export_schemas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_contract_is_up_to_date():
    mod = _load()
    stale = [str(p.relative_to(REPO)) for p, text in mod.outputs().items() if not p.is_file() or p.read_text() != text]
    assert not stale, f"run `python scripts/export_schemas.py`: {stale}"


def test_openapi_covers_every_spec_endpoint():
    spec = json.loads((REPO / "docs" / "openapi.json").read_text())
    have = {(m.upper(), p) for p, ops in spec["paths"].items() for m in ops}
    want = {
        ("GET", "/api/health"), ("GET", "/api/levels"), ("GET", "/api/maps"),
        ("POST", "/api/sessions"), ("POST", "/api/sessions/{session_id}/reset"),
        ("POST", "/api/sessions/{session_id}/step"), ("GET", "/api/sessions/{session_id}"),
        ("DELETE", "/api/sessions/{session_id}"), ("GET", "/api/models"), ("GET", "/api/models/{model_id}"),
        ("GET", "/api/experiments"), ("GET", "/api/experiments/{run_id}"),
        ("GET", "/api/experiments/{run_id}/metrics"), ("GET", "/api/experiments/{run_id}/eval"),
        ("GET", "/api/studies"), ("GET", "/api/studies/{study_id}"), ("GET", "/api/replays"),
        ("GET", "/api/replays/{replay_id}"), ("GET", "/api/analysis/{study_id}/heatmap"),
        ("POST", "/api/training/jobs"), ("GET", "/api/training/jobs/{job_id}"),
        ("POST", "/api/training/jobs/{job_id}/cancel"),
    }
    assert want <= have, want - have


def test_replay_schema_exported():
    s = json.loads((REPO / "docs" / "schemas" / "replay.schema.json").read_text())
    assert s["properties"]["schema_version"]["default"] == 1 and "frames" in s["properties"]

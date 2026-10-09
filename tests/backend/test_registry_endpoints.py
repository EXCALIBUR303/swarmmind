"""Read-only endpoints: levels, maps, models, experiments, studies, analysis, replays (§28.2)."""

from __future__ import annotations

import gzip
import json

from tests.backend.conftest import REPLAY_ID, RUN_ID


def test_levels_lists_all_four(client):
    body = client.get("/api/levels").json()
    assert [lv["level"] for lv in body] == ["L1", "L2", "L3", "L4"]
    assert body[1]["config"]["n_robots"] == 3


def test_maps_validation_is_open_and_complete(client):
    maps = client.get("/api/maps", params={"set": "validation", "level": "L2"}).json()
    assert len(maps) == 50
    m = maps[0]
    assert len(m["terrain"]) == m["height"] and m["set"] == "validation" and m["n_robots"] == 3
    assert len(m["hash"]) > 8


def test_maps_heldout_goes_through_gatekeeper_and_logs(client, results_tree):
    log = results_tree / "results" / "heldout_access_log.jsonl"
    assert not log.exists()
    maps = client.get("/api/maps", params={"set": "structured", "level": "L2"}).json()
    assert {m["id"] for m in maps} >= {"L2-struct-bottleneck_vertical"}
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["split"] == "structured" and rec["who"] == "backend_api"
    assert client.get("/api/maps", params={"set": "validation"}).status_code == 200
    # validation is open: still just the one held-out access line so far
    assert len(log.read_text().splitlines()) == 1


def test_maps_validation_errors(client):
    assert client.get("/api/maps", params={"set": "secret"}).status_code == 422
    assert client.get("/api/maps", params={"set": "test", "level": "L9"}).status_code == 404
    assert client.get("/api/maps", params={"set": "validation", "level": "../x"}).status_code == 422


def test_models_list_get_and_filters(client):
    models = client.get("/api/models").json()
    assert [m["model_id"] for m in models] == [RUN_ID]  # running / corrupt runs are not offered
    m = client.get(f"/api/models/{RUN_ID}").json()
    assert m["algo"] == "ippo" and m["level"] == "L1" and m["test_metrics"] == {"success_rate": 0.4}
    assert client.get("/api/models", params={"algo": "mappo"}).json() == []
    assert client.get("/api/models/nope").status_code == 404


def test_experiments_list_filter_and_invalid_runs(client):
    runs = client.get("/api/experiments").json()
    by_id = {r["run_id"]: r for r in runs}
    assert by_id["s2-corrupt"]["status"] == "invalid"  # tolerated, never raised
    assert by_id[RUN_ID]["has_actor"] is True
    assert [r["run_id"] for r in client.get("/api/experiments", params={"status": "running"}).json()] == ["s1-20260101T000000Z-bbbb2222"]
    assert len(client.get("/api/experiments", params={"algo": "ippo", "level": "L1"}).json()) == 2


def test_experiment_detail_hides_paths(client):
    body = client.get(f"/api/experiments/{RUN_ID}").json()
    assert body["summary"]["status"] == "completed"
    assert "checkpoints" not in body["manifest"] and "versions" not in body["manifest"]
    assert "/abs/secret" not in json.dumps(body)
    assert client.get("/api/experiments/s9-missing").status_code == 404


def test_experiment_metrics_from_update_and_eval(client):
    all_rows = client.get(f"/api/experiments/{RUN_ID}/metrics").json()
    assert [r["update"] for r in all_rows] == [0, 1, 2]
    assert [r["update"] for r in client.get(f"/api/experiments/{RUN_ID}/metrics", params={"from_update": 2}).json()] == [2]
    assert client.get(f"/api/experiments/{RUN_ID}/metrics", params={"from_update": -1}).status_code == 422
    ev = client.get(f"/api/experiments/{RUN_ID}/eval").json()
    assert ev["validation_history"][0]["success_rate"] == 0.5
    assert "episodes" not in ev["test_L1"]["modes"]["stochastic"]
    full = client.get(f"/api/experiments/{RUN_ID}/eval", params={"episodes": True}).json()
    assert full["test_L1"]["modes"]["stochastic"]["episodes"] == [{"x": 1}]


def test_studies_and_heatmap(client):
    studies = client.get("/api/studies").json()
    assert [s["study"] for s in studies] == ["demo"]
    detail = client.get("/api/studies/demo").json()
    assert detail["manifest"]["n_jobs"] == 2 and "git" not in detail["manifest"]
    assert detail["analysis_files"] == ["L1_ippo_L1-m1_heatmap.json"]
    hm = client.get("/api/analysis/demo/heatmap", params={"map_id": "L1-m1", "algo": "ippo", "level": "L1", "kind": "collisions"}).json()
    assert hm["counts"] == [[1]] and hm["kind"] == "collisions"
    assert client.get("/api/analysis/demo/heatmap", params={"map_id": "L1-m1", "algo": "ippo", "level": "L1", "kind": "bad"}).status_code == 422
    assert client.get("/api/analysis/demo/heatmap", params={"map_id": "zzz", "algo": "ippo", "level": "L1"}).status_code == 404
    assert client.get("/api/studies/missing").status_code == 404


def test_replays_list_filter_and_gzip_passthrough(client, results_tree):
    items = client.get("/api/replays").json()
    assert [i["replay_id"] for i in items] == [REPLAY_ID] and items[0]["run_id"] == RUN_ID
    assert client.get("/api/replays", params={"map_id": "nope"}).json() == []
    r = client.get(f"/api/replays/{REPLAY_ID}")
    assert r.status_code == 200 and r.headers["content-encoding"] == "gzip"
    on_disk = gzip.decompress((results_tree / "results" / "replays" / "demo" / RUN_ID / f"{REPLAY_ID}.json.gz").read_bytes())
    assert r.content == on_disk  # byte-identical after transport decoding
    assert r.json()["schema_version"] == 1 and r.json()["frames"]
    assert client.get("/api/replays/missing").status_code == 404


def test_replay_initial_recorded_and_derived_agree(client, results_tree):
    """New replays carry `initial`; old ones get it derived by re-running reset. Same answer."""
    from marl.replay.loader import load_replay, save_replay

    path = results_tree / "results" / "replays" / "demo" / RUN_ID / f"{REPLAY_ID}.json.gz"
    recorded = client.get(f"/api/replays/{REPLAY_ID}/initial").json()
    assert recorded["source"] == "recorded" and recorded["known_team"]

    old_style = load_replay(path).model_copy(update={"initial": None})
    save_replay(old_style, path)
    derived = client.get(f"/api/replays/{REPLAY_ID}/initial").json()
    assert derived["source"] == "derived"
    assert sorted(map(tuple, derived["known_team"])) == sorted(map(tuple, recorded["known_team"]))
    assert client.get("/api/replays/missing/initial").status_code == 404


def test_analysis_file_endpoint(client):
    assert client.get("/api/studies/demo/analysis/L1_ippo_L1-m1_heatmap").json()["visit_counts"] == [[3]]
    assert client.get("/api/studies/demo/analysis/nope").status_code == 404
    assert client.get("/api/studies/demo/analysis/%2e%2e").status_code in (404, 422)
    assert client.get("/api/studies/missing/analysis/x").status_code == 404


def test_preregistration_rules_are_quoted_from_the_frozen_doc(client):
    body = client.get("/api/preregistration").json()
    ids = [r["id"] for r in body["rules"]]
    assert ids[:4] == ["H1", "H2", "H3", "H4"]
    doc = (__import__("pathlib").Path(__file__).resolve().parents[2] / "docs" / "EXPERIMENTS.md").read_text()
    for r in body["rules"]:
        assert r["text"] in doc  # verbatim substring, not a paraphrase

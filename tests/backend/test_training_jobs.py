"""Training job API + subprocess runner (§28.2, §28.5, §37).

The launch/cancel tests start the REAL `scripts/train.py` as a subprocess on
the L1 level with a tiny budget, writing into a tmp runs dir."""

from __future__ import annotations

import time

import pytest

from backend.app.schemas.training import ALLOWED_OVERRIDES, TrainRequest

REQ = {"level": "L1", "algo": "ippo", "seed": 0, "total_env_steps": 2048,
       "overrides": {"ppo.n_envs": 4, "ppo.rollout_len": 64, "ppo.epochs": 1}}


@pytest.mark.parametrize(
    "patch",
    [
        {"level": "L9"},
        {"algo": "a2c"},
        {"seed": -1},
        {"total_env_steps": 100},
        {"total_env_steps": 10_000_001},
        {"overrides": {"device": "cuda"}},
        {"overrides": {"ppo.lr_actor": 5.0}},
        {"overrides": {"ppo.n_envs": 2.5}},
        {"overrides": {"ppo.epochs": 0}},
        {"overrides": {"ppo.lr_actor; rm -rf /": 1}},
        {"overrides": {"ppo.lr_actor": True}},
    ],
)
def test_launch_validation_rejects(client, patch):
    r = client.post("/api/training/jobs", json={**REQ, **patch})
    assert r.status_code == 422, (patch, r.text)
    assert client.get("/api/training/jobs").json() == []  # nothing was started


def test_ui_step_limit_follows_settings(make_client):
    c = make_client(max_ui_train_env_steps=4096)
    assert c.post("/api/training/jobs", json={**REQ, "total_env_steps": 8192}).status_code == 422


def test_allowlist_is_small_and_numeric():
    assert all(t in (int, float) for t, _, _ in ALLOWED_OVERRIDES.values())
    TrainRequest.model_validate(REQ)


def test_unknown_job_404(client):
    assert client.get("/api/training/jobs/nope").status_code == 404
    assert client.post("/api/training/jobs/nope/cancel").status_code == 404
    assert client.get("/api/training/jobs/..").status_code in (404, 422)


@pytest.mark.slow
@pytest.mark.integration
def test_launch_runs_real_trainer_to_completion(client):
    r = client.post("/api/training/jobs", json=REQ)
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] in ("starting", "running") and job["request"]["level"] == "L1"

    seen = {job["status"]}
    deadline = time.monotonic() + 240
    while job["status"] in ("starting", "running") and time.monotonic() < deadline:
        time.sleep(0.5)
        job = client.get(f"/api/training/jobs/{job['job_id']}").json()
        seen.add(job["status"])
    assert job["status"] == "completed", job
    assert job["exit_code"] == 0 and job["run_status"] == "completed" and job["run_id"]
    assert seen <= {"starting", "running", "completed"}

    run = client.get(f"/api/experiments/{job['run_id']}").json()
    assert run["summary"]["study"] == "ui" and run["summary"]["status"] == "completed"
    assert client.get(f"/api/experiments/{job['run_id']}/metrics").json(), "metrics.jsonl was written"
    assert any(m["model_id"] == job["run_id"] for m in client.get("/api/models").json()), "finished run is now a model"
    assert [j["job_id"] for j in client.get("/api/training/jobs").json()] == [job["job_id"]]
    assert client.post(f"/api/training/jobs/{job['job_id']}/cancel").status_code == 409  # already finished


@pytest.mark.slow
@pytest.mark.integration
def test_one_job_at_a_time_and_cancel_marks_run_interrupted(client):
    big = {**REQ, "total_env_steps": 2_000_000}
    first = client.post("/api/training/jobs", json=big)
    assert first.status_code == 202
    jid = first.json()["job_id"]
    try:
        assert client.post("/api/training/jobs", json=big).status_code == 409  # max 1 concurrent

        deadline = time.monotonic() + 120
        job = first.json()
        while job["status"] != "running" and time.monotonic() < deadline:
            time.sleep(0.5)
            job = client.get(f"/api/training/jobs/{jid}").json()
        assert job["status"] == "running" and job["run_id"], job

        cancelled = client.post(f"/api/training/jobs/{jid}/cancel")
        assert cancelled.status_code == 200
        deadline = time.monotonic() + 60
        while job["status"] in ("starting", "running") and time.monotonic() < deadline:
            time.sleep(0.5)
            job = client.get(f"/api/training/jobs/{jid}").json()
        assert job["status"] == "cancelled", job
        assert job["run_status"] == "interrupted"  # trainer handled SIGTERM and wrote its manifest
    finally:
        client.post(f"/api/training/jobs/{jid}/cancel")
    # slot is free again
    assert client.post("/api/training/jobs", json=big).status_code == 202
    last = client.get("/api/training/jobs").json()
    for j in last:
        client.post(f"/api/training/jobs/{j['job_id']}/cancel")


def test_recycled_pid_is_not_mistaken_for_the_training_process(make_client, results_tree):
    """After an API restart the runner only has a pid. If that pid now belongs to a
    different process (start-time token differs) the job must not look alive, or the
    single job slot would be blocked forever and cancel would signal a stranger."""
    import json
    import os

    c = make_client()
    jobs = results_tree / "results" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    rec = {"job_id": "stale0001", "request": REQ, "pid": os.getpid(), "pid_token": "Thu Jan  1 00:00:00 1970",
           "run_id": None, "created_at": "2026-01-01T00:00:00+00:00", "finished_at": None, "exit_code": None, "cancelled": False}
    (jobs / "stale0001.json").write_text(json.dumps(rec))
    assert c.get("/api/training/jobs/stale0001").json()["status"] == "failed"
    assert c.post("/api/training/jobs/stale0001/cancel").status_code == 409  # never signals the stranger

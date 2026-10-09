"""Launches `scripts/train.py` as a subprocess (§28.5, §37).

Never trains inside the API process; never uses a shell; argv is built
only from validated, allowlisted values. Status is derived from the
process and the run's own manifest."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.schemas.training import JobStatus, TrainRequest
from backend.app.services.errors import ConflictError, NotFoundError, ValidationFailed
from backend.app.services.jsonio import read_json, read_jsonl
from marl.algorithms.common.config import AlgoConfig
from marl.utils.config_loader import apply_overrides, load_yaml
from marl.utils.paths import CONFIGS_DIR, REPO_ROOT, is_safe_id

UI_STUDY = "ui"
KILL_GRACE_SECONDS = 15.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists, not ours
        return True
    # A zombie child still "exists"; reap it if it is ours.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def _start_token(pid: int) -> str | None:
    """Process start time (`ps -o lstart=`), so a recycled pid is never mistaken
    for the training process that was launched earlier."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, integer pid
            ["ps", "-o", "lstart=", "-p", str(int(pid))], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


class JobRunner:
    def __init__(self, train_script: Path, runs_dir: Path, jobs_dir: Path, max_concurrent: int, max_env_steps: int) -> None:
        self.train_script = train_script
        self.runs_dir = runs_dir
        self.jobs_dir = jobs_dir
        self.max_concurrent = max_concurrent
        self.max_env_steps = max_env_steps
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._cancel_requested: dict[str, float] = {}

    # -- persistence ----------------------------------------------------
    def _record_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _write(self, rec: dict[str, Any]) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        path = self._record_path(rec["job_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2))
        tmp.replace(path)

    def _read(self, job_id: str) -> dict[str, Any]:
        if not is_safe_id(job_id):
            raise NotFoundError(f"job '{job_id}' not found")
        rec = read_json(self._record_path(job_id))
        if not isinstance(rec, dict):
            raise NotFoundError(f"job '{job_id}' not found")
        return rec

    # -- validation -----------------------------------------------------
    def _argv(self, req: TrainRequest) -> list[str]:
        if req.total_env_steps > self.max_env_steps:
            raise ValidationFailed(f"total_env_steps exceeds the UI limit of {self.max_env_steps}")
        overrides = [f"{k}={v}" for k, v in sorted(req.overrides.items())]
        # Same schema the CLI uses; surfaces e.g. minibatch/rollout inconsistencies now.
        try:
            raw = apply_overrides(load_yaml(CONFIGS_DIR / "algo" / f"{req.algo}.yaml"), overrides)
            AlgoConfig.model_validate(raw)
        except (KeyError, ValueError) as exc:
            raise ValidationFailed(f"invalid training parameters: {exc}") from exc
        argv = [
            sys.executable, str(self.train_script),
            "--level", req.level, "--algo", req.algo, "--seed", str(req.seed),
            "--total-env-steps", str(req.total_env_steps),
            "--study", UI_STUDY, "--runs-dir", str(self.runs_dir),
        ]
        if overrides:
            argv += ["--set", *overrides]
        return argv

    # -- API ------------------------------------------------------------
    def active_count(self) -> int:
        return sum(1 for j in self.list() if j.status in ("starting", "running"))

    def submit(self, req: TrainRequest) -> JobStatus:
        argv = self._argv(req)
        with self._lock:
            if self.active_count() >= self.max_concurrent:
                raise ConflictError(f"at most {self.max_concurrent} training job(s) may run at once")
            job_id = uuid.uuid4().hex[:16]
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.jobs_dir / f"{job_id}.log"
            with log_path.open("wb") as log:
                proc = subprocess.Popen(  # noqa: S603 - argv list, fixed script, validated args
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=REPO_ROOT,
                    start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            self._procs[job_id] = proc
            self._write(
                {
                    "job_id": job_id,
                    "request": req.model_dump(),
                    "pid": proc.pid,
                    "pid_token": _start_token(proc.pid),
                    "run_id": None,
                    "created_at": datetime.now(UTC).isoformat(),
                    "finished_at": None,
                    "exit_code": None,
                    "cancelled": False,
                }
            )
        return self.status(job_id)

    def _find_run(self, rec: dict[str, Any]) -> Path | None:
        if rec.get("run_id"):
            for p in self.runs_dir.glob(f"{UI_STUDY}/*/*/{rec['run_id']}"):
                return p
            return None
        log = self.jobs_dir / f"{rec['job_id']}.log"
        try:
            for line in log.read_text(errors="replace").splitlines()[:20]:
                if line.startswith("run_dir:"):
                    p = Path(line.split(":", 1)[1].strip()).resolve()
                    if p.is_dir() and p.is_relative_to(self.runs_dir.resolve()) and is_safe_id(p.name):
                        rec["run_id"] = p.name
                        return p
        except OSError:
            pass
        return None

    def status(self, job_id: str) -> JobStatus:
        rec = self._read(job_id)
        pid = int(rec["pid"])
        proc = self._procs.get(job_id)
        exit_code = rec.get("exit_code")
        if proc is not None and proc.poll() is not None:
            exit_code = proc.returncode
        if proc is not None:
            alive = proc.poll() is None
        else:  # after an API restart: alive only if the SAME process (not a recycled pid) still runs
            alive = exit_code is None and _pid_alive(pid) and _start_token(pid) == rec.get("pid_token")

        # Escalate a cancel that was ignored.
        asked = self._cancel_requested.get(job_id)
        if alive and asked is not None and time.monotonic() - asked > KILL_GRACE_SECONDS:
            self._signal(pid, signal.SIGKILL)

        run_dir = self._find_run(rec)
        manifest = read_json(run_dir / "manifest.json") if run_dir else None
        run_status = manifest.get("status") if isinstance(manifest, dict) else None

        if alive:
            status = "running" if run_status == "running" else "starting"
        elif rec.get("cancelled"):
            status = "cancelled"
        elif exit_code == 0 and run_status == "completed":
            status = "completed"
        else:
            status = "failed"

        if not alive and rec.get("finished_at") is None:
            rec["finished_at"] = datetime.now(UTC).isoformat()
            rec["exit_code"] = exit_code
            self._write(rec)
        elif rec.get("run_id") and not rec.get("_saved_run"):
            rec["_saved_run"] = True
            self._write(rec)

        progress = None
        if run_dir:
            rows = read_jsonl(run_dir / "metrics.jsonl")
            if rows:
                last = rows[-1]
                progress = {"update": last.get("update"), "env_steps": last.get("env_steps"),
                            "total_env_steps": rec["request"]["total_env_steps"]}
        return JobStatus(
            job_id=job_id,
            status=status,
            run_id=rec.get("run_id"),
            request=TrainRequest.model_validate(rec["request"]),
            created_at=rec["created_at"],
            finished_at=rec.get("finished_at"),
            exit_code=exit_code,
            run_status=run_status,
            progress=progress,
        )

    @staticmethod
    def _signal(pid: int, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, sig)  # start_new_session=True made pid its own group

    def cancel(self, job_id: str) -> JobStatus:
        rec = self._read(job_id)
        st = self.status(job_id)
        if st.status not in ("starting", "running"):
            raise ConflictError(f"job is already {st.status}")
        rec["cancelled"] = True
        self._write(rec)
        self._cancel_requested[job_id] = time.monotonic()
        self._signal(int(rec["pid"]), signal.SIGTERM)  # trainer marks the run 'interrupted'
        return self.status(job_id)

    def list(self) -> list[JobStatus]:
        out: list[JobStatus] = []
        if not self.jobs_dir.is_dir():
            return out
        for f in sorted(self.jobs_dir.glob("*.json")):
            if is_safe_id(f.stem):
                try:
                    out.append(self.status(f.stem))
                except (NotFoundError, KeyError, ValueError):
                    continue
        return out

    def wait(self, job_id: str, timeout: float) -> JobStatus:
        """Test/ops helper: block until the job leaves starting/running."""
        end = time.monotonic() + timeout
        st = self.status(job_id)
        while st.status in ("starting", "running") and time.monotonic() < end:
            time.sleep(0.25)
            st = self.status(job_id)
        return st

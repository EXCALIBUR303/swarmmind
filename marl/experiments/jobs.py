"""Study job planning and the idempotency rule: which runs exist, which
are finished, which can be resumed, and which a live process owns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marl.envs.core.config import LevelRegistry
from marl.experiments.config import StudyConfig

# ---------------------------------------------------------------- jobs


@dataclass(frozen=True)
class TrainJob:
    study: str
    algo: str
    level: str
    seed: int
    total_env_steps: int

    @property
    def key(self) -> str:
        return f"train/{self.algo}/{self.level}/s{self.seed}"


def plan_train_jobs(cfg: StudyConfig) -> list[TrainJob]:
    """Deterministic job order: level, then seed, then algo — so IPPO and
    MAPPO of the same seed run side by side and partial studies stay paired."""
    return [
        TrainJob(cfg.study, algo, level, seed, cfg.total_env_steps)
        for level in cfg.levels
        for seed in cfg.seeds
        for algo in cfg.algos
    ]


def resolved_run_config(cfg: StudyConfig, algo: str, level: str, seed: int) -> dict[str, Any]:
    """Exactly the dict the trainer hashes into `config_hash` (§32)."""
    env_cfg = LevelRegistry().get(level)
    return {"env": env_cfg.model_dump(), "algo": cfg.resolve_algo(algo).model_dump(), "run_seed": seed}


def find_run_dirs(job: TrainJob, h8: str, runs_root: Path) -> list[Path]:
    parent = runs_root / job.study / job.algo / job.level
    return sorted(parent.glob(f"s{job.seed}-*-{h8}")) if parent.is_dir() else []


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_is_locked(run_dir: Path) -> bool:
    lock = run_dir / ".lock"
    if not lock.is_file():
        return False
    try:
        pid = int(lock.read_text().strip())
    except ValueError:
        return False
    return pid != os.getpid() and _pid_alive(pid)


@dataclass
class RunState:
    action: str  # "skip" | "resume" | "new" | "busy"
    run_dir: Path | None
    reason: str = ""


def classify_run(job: TrainJob, h8: str, runs_root: Path) -> RunState:
    """Decide what to do for one training job (the idempotency rule)."""
    candidates = find_run_dirs(job, h8, runs_root)
    for d in candidates:
        m = _read_json(d / "manifest.json") or {}
        if m.get("status") == "completed":
            budget = m.get("budget_env_steps")
            if budget != job.total_env_steps:
                raise RuntimeError(
                    f"{d} is completed with budget {budget}, but the study asks for "
                    f"{job.total_env_steps}; refusing to mix budgets in one study"
                )
            return RunState("skip", d, "completed")
    for d in reversed(candidates):  # newest first
        if run_is_locked(d):
            return RunState("busy", d, "another live process holds its lock")
        if (d / "checkpoints" / "latest.pt").is_file():
            return RunState("resume", d, f"status={(_read_json(d / 'manifest.json') or {}).get('status')}")
    return RunState("new", None, "no resumable run")

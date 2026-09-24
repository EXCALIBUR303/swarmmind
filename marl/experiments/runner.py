"""Study orchestration (Phase 6 task 1-2): levels × algos × seeds → runs,
then final held-out evaluation and baselines, all idempotent.

- Every training run executes in its own `spawn`ed process with ONE torch
  thread (§21.5; measured in Phase 6: intra-op threads make these small
  CNNs slower, and `fork` + torch is deadlock-prone on macOS).
- Re-running a study skips completed runs whose budget matches, resumes
  interrupted/failed ones from `checkpoints/latest.pt`, and re-evaluates a
  run only if its eval file is missing or was written under another
  protocol. A run a live process is working on (per its `.lock` pid) is
  never touched.
- The main process is the only writer of `study_manifest.json`.
"""

from __future__ import annotations

import json
import subprocess
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from marl.experiments.config import StudyConfig, study_config_path
from marl.experiments.jobs import (  # noqa: F401  (re-exported: the runner's public API)
    RunState,
    TrainJob,
    _read_json,
    classify_run,
    find_run_dirs,
    plan_train_jobs,
    resolved_run_config,
    run_is_locked,
)
from marl.experiments.workers import (  # noqa: F401
    _eval_key,
    _raise_on_sigterm,
    _worker_init,
    baseline_worker,
    eval_worker,
    train_worker,
)
from marl.utils.hashing import config_hash, sha256_hex
from marl.utils.paths import CONFIGS_DIR, MAPS_DIR, REPO_ROOT, RUNS_DIR, STUDIES_DIR

FROZEN_FILE = CONFIGS_DIR / "experiments" / "FROZEN.json"


# ------------------------------------------------------ frozen inputs


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_inputs(cfg: StudyConfig, study_yaml: Path) -> dict[str, str]:
    """sha256 of every file that defines the study's experimental design."""
    files = [study_yaml, CONFIGS_DIR / "algo" / "ippo.yaml", CONFIGS_DIR / "algo" / "mappo.yaml",
             MAPS_DIR / "manifest.json"]
    files += [CONFIGS_DIR / "env" / f"{lvl}.yaml" for lvl in cfg.levels]
    def key(p: Path) -> str:
        return str(p.relative_to(REPO_ROOT)) if p.resolve().is_relative_to(REPO_ROOT) else str(p)

    return {key(p): _sha256_file(p) for p in files}


def check_frozen(study: str, current: dict[str, str], frozen_file: Path = FROZEN_FILE) -> list[str]:
    """Deviations of `current` from the pre-registered hashes; [] if the
    study was never frozen or nothing changed."""
    registry = _read_json(frozen_file) or {}
    frozen = registry.get(study, {}).get("files")
    if not frozen:
        return []
    out = [f"{p}: changed since freeze" for p, h in frozen.items() if current.get(p) not in (None, h)]
    out += [f"{p}: missing from current inputs" for p in frozen if p not in current]
    return out


def freeze_study(cfg: StudyConfig, study_yaml: Path, frozen_file: Path = FROZEN_FILE) -> dict[str, Any]:
    registry = _read_json(frozen_file) or {}
    entry = {"frozen_at": datetime.now(UTC).isoformat(), "git": git_state(), "files": frozen_inputs(cfg, study_yaml)}
    registry[cfg.study] = entry
    _atomic_json(frozen_file, registry)
    return entry


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")
    tmp.replace(path)


# ------------------------------------------------------------ the study


class StudyManifest:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path, self.data = path, data

    def job(self, key: str, **fields: Any) -> None:
        entry = self.data["jobs"].setdefault(key, {"attempts": 0})
        entry.update(fields)
        entry["updated"] = datetime.now(UTC).isoformat()
        self.save()

    def save(self) -> None:
        self.data["updated"] = datetime.now(UTC).isoformat()
        _atomic_json(self.path, self.data)


def run_study(
    cfg: StudyConfig,
    parallel: int = 1,
    runs_root: Path = RUNS_DIR,
    studies_root: Path = STUDIES_DIR,
    stages: tuple[str, ...] = ("train", "eval", "baselines"),
    allow_deviation: bool = False,
    study_yaml: Path | None = None,
    max_eval_maps: int | None = None,
    access_log: Path | None = None,
    frozen_file: Path = FROZEN_FILE,
    log: Callable[[str], None] = print,
) -> StudyManifest:
    """Run (or continue) a whole study. `parallel=0` runs every job inline
    in this process (tests, debugging); `parallel>=1` uses a spawn pool."""
    study_yaml = study_yaml or study_config_path(cfg.study)
    inputs = frozen_inputs(cfg, study_yaml)
    deviations = check_frozen(cfg.study, inputs, frozen_file)
    if deviations and not allow_deviation:
        raise RuntimeError(f"study '{cfg.study}' is frozen and its inputs changed: {deviations}")

    path = studies_root / cfg.study / "study_manifest.json"
    data = _read_json(path) or {"study": cfg.study, "created": datetime.now(UTC).isoformat(), "jobs": {}}
    data.pop("finished", None)
    data.update(
        started=datetime.now(UTC).isoformat(),
        code_fingerprint=study_code_fingerprint(),
        config=cfg.model_dump(),
        smoke=cfg.smoke,
        inputs_sha256=inputs,
        git=git_state(),
        deviations=sorted(set(data.get("deviations", [])) | set(deviations)),
        stages=list(stages),
        parallel=parallel,
    )
    manifest = StudyManifest(path, data)
    manifest.save()
    cfg_dict = cfg.model_dump()
    access = str(access_log) if access_log else None

    pool = ProcessPoolExecutor(max_workers=parallel, mp_context=get_context("spawn"),
                               initializer=_worker_init) if parallel > 0 else None
    pending: dict[Future, str] = {}

    def submit(key: str, fn: Callable[..., dict[str, Any]], *args: Any) -> None:
        entry = manifest.data["jobs"].setdefault(key, {"attempts": 0})
        manifest.job(key, status="running", attempts=entry["attempts"] + 1)
        if pool is None:
            _finish(key, fn(*args))
        else:
            pending[pool.submit(fn, *args)] = key

    def _finish(key: str, rec: dict[str, Any]) -> None:
        manifest.job(key, **{k: v for k, v in rec.items() if k != "key"})
        log(f"[{rec['status']:>9}] {key}" + (f"  ({rec.get('action')})" if rec.get("action") else ""))
        if rec.get("error"):
            log(rec["error"])
        if key.startswith("train/") and rec["status"] == "completed" and "eval" in stages:
            submit(_eval_key(Path(rec["run_dir"])), eval_worker, cfg_dict, rec["run_dir"], max_eval_maps, access)

    def drain() -> None:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                key = pending.pop(fut)
                try:
                    rec = fut.result()
                except Exception:  # the worker PROCESS died (e.g. OOM-killed); workers never raise
                    rec = {"key": key, "status": "failed", "error": traceback.format_exc(limit=4)}
                _finish(key, rec)

    try:
        if "train" in stages:
            for job in plan_train_jobs(cfg):
                submit(job.key, train_worker, cfg_dict, job.__dict__, str(runs_root))
        elif "eval" in stages:  # eval-only pass over already-completed runs
            for job in plan_train_jobs(cfg):
                h8 = config_hash(resolved_run_config(cfg, job.algo, job.level, job.seed))
                state = classify_run(job, h8, runs_root)
                if state.action == "skip" and state.run_dir is not None:
                    submit(_eval_key(state.run_dir), eval_worker, cfg_dict, str(state.run_dir), max_eval_maps, access)
        drain()
        # Baselines touch held-out maps too, so they wait until training is over.
        if "baselines" in stages:
            for level in cfg.levels:
                for s in cfg.baselines.random.eval_seeds:
                    submit(f"baseline/random/{level}/s{s}", baseline_worker, cfg_dict, "random", level, s,
                           str(studies_root), max_eval_maps, access)
                if cfg.baselines.heuristic is not None:
                    submit(f"baseline/heuristic/{level}/ref", baseline_worker, cfg_dict, "heuristic", level, None,
                           str(studies_root), max_eval_maps, access)
            drain()
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    manifest.data["finished"] = datetime.now(UTC).isoformat()
    manifest.save()
    return manifest


def study_code_fingerprint() -> str:
    """Short hash of the marl/ source tree (recorded for provenance)."""
    h = []
    for p in sorted((REPO_ROOT / "marl").rglob("*.py")):
        h.append(f"{p.relative_to(REPO_ROOT)}:{_sha256_file(p)}")
    return sha256_hex("\n".join(h))[:12]

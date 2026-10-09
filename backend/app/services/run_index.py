"""Indexes `results/runs/<study>/<algo>/<level>/<run_id>/` (§28.1).

Tolerates partial or corrupt runs: a run whose manifest is missing or
unparseable is listed with `status="invalid"`, never raised."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from backend.app.services.errors import NotFoundError
from backend.app.services.jsonio import read_json, read_jsonl, scrub
from marl.utils.paths import is_safe_id


class RunIndex:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._lock = threading.Lock()
        self._runs: dict[str, Path] = {}

    def refresh(self) -> None:
        by_name: dict[str, list[Path]] = {}
        if self.runs_dir.is_dir():
            for study in sorted(self.runs_dir.iterdir()):
                for algo in sorted(study.iterdir()) if study.is_dir() else []:
                    for level in sorted(algo.iterdir()) if algo.is_dir() else []:
                        for run in sorted(level.iterdir()) if level.is_dir() else []:
                            if run.is_dir() and is_safe_id(run.name):
                                by_name.setdefault(run.name, []).append(run)
        # Pipeline run ids (`s0-<timestamp>-<hash>`) are unique. Early pilot
        # runs were all named `seed0`; on a collision every colliding run is
        # addressed as `<study>.<algo>.<level>.<name>` so none is hidden.
        found: dict[str, Path] = {}
        for name, paths in by_name.items():
            if len(paths) == 1:
                found[name] = paths[0]
            else:
                for p in paths:
                    found[f"{p.parent.parent.parent.name}.{p.parent.parent.name}.{p.parent.name}.{name}"] = p
        with self._lock:
            self._runs = found

    def paths(self) -> dict[str, Path]:
        self.refresh()
        with self._lock:
            return dict(self._runs)

    def _dir(self, run_id: str) -> Path:
        with self._lock:
            path = self._runs.get(run_id)
        if path is None or not path.is_dir():
            self.refresh()  # a run may have been created since the last scan
            with self._lock:
                path = self._runs.get(run_id)
        if path is None or not path.is_dir():
            raise NotFoundError(f"run '{run_id}' not found")
        return path

    @staticmethod
    def _summary(run_id: str, path: Path) -> dict[str, Any]:
        manifest = read_json(path / "manifest.json")
        rel = path.parent.parent.parent.name, path.parent.parent.name, path.parent.name
        if not isinstance(manifest, dict):
            return {"run_id": run_id, "study": rel[0], "algo": rel[1], "level": rel[2], "status": "invalid"}
        return {
            "run_id": run_id,
            "study": manifest.get("study", rel[0]),
            "algo": manifest.get("algo", rel[1]),
            "level": manifest.get("level", rel[2]),
            "seed": manifest.get("seed"),
            "status": manifest.get("status", "invalid"),
            "total_env_steps": manifest.get("total_env_steps"),
            "trained_env_steps": manifest.get("trained_env_steps"),
            "budget_env_steps": manifest.get("budget_env_steps"),
            "updates": manifest.get("updates"),
            "start_time": manifest.get("start_time"),
            "end_time": manifest.get("end_time"),
            "wall_time_seconds": manifest.get("wall_time_seconds"),
            "final_val_metrics": manifest.get("final_val_metrics"),
            "has_actor": (path / "actor.pt").is_file(),
        }

    def list_runs(
        self,
        study: str | None = None,
        algo: str | None = None,
        level: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            runs = dict(self._runs)
        out = [self._summary(rid, p) for rid, p in runs.items()]
        for key, want in (("study", study), ("algo", algo), ("level", level), ("status", status)):
            if want is not None:
                out = [r for r in out if r.get(key) == want]
        out.sort(key=lambda r: (str(r.get("start_time") or ""), r["run_id"]), reverse=True)
        return out

    def get(self, run_id: str) -> dict[str, Any]:
        path = self._dir(run_id)
        manifest = read_json(path / "manifest.json")
        summary = self._summary(run_id, path)
        if isinstance(manifest, dict):
            # Never expose absolute filesystem paths.
            manifest = {k: v for k, v in manifest.items() if k not in ("checkpoints", "versions")}
            manifest["n_checkpoints"] = len(list((path / "checkpoints").glob("step_*.pt"))) if (path / "checkpoints").is_dir() else 0
        return scrub({"summary": summary, "manifest": manifest})

    def metrics(self, run_id: str, from_update: int = 0) -> list[dict[str, Any]]:
        rows = read_jsonl(self._dir(run_id) / "metrics.jsonl")
        return [r for r in rows if int(r.get("update", 0)) >= from_update]

    def val_evals(self, run_id: str, from_update: int = -1) -> list[dict[str, Any]]:
        rows = read_jsonl(self._dir(run_id) / "eval_val.jsonl")
        return [r for r in rows if int(r.get("update", 0)) >= from_update]

    def eval_results(self, run_id: str, include_episodes: bool = False) -> dict[str, Any]:
        """Validation history plus every `eval/*.json` file. Per-episode
        records are large (hundreds per file) so they are opt-in."""
        eval_dir = self._dir(run_id) / "eval"
        out: dict[str, Any] = {"validation_history": self.val_evals(run_id)}
        if eval_dir.is_dir():
            for f in sorted(eval_dir.glob("*.json")):
                data = read_json(f)
                if data is None:
                    continue
                if not include_episodes and isinstance(data.get("modes"), dict):
                    data = {
                        **data,
                        "modes": {
                            m: {k: v for k, v in body.items() if k != "episodes"}
                            for m, body in data["modes"].items()
                            if isinstance(body, dict)
                        },
                    }
                out[f.stem] = data
        return scrub(out)

    def run_dir(self, run_id: str) -> Path:
        return self._dir(run_id)

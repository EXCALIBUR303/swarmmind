"""Discovers loadable actors: `results/runs/**/actor.pt` and
`models/<name>/actor.pt` (§28.1). Actor-only files, loaded with
`weights_only=True` through `marl`'s `load_actor` — never a critic."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.services.errors import NotFoundError, ValidationFailed
from backend.app.services.jsonio import read_json, scrub
from backend.app.services.run_index import RunIndex
from marl.utils.paths import is_safe_id


@dataclass
class ModelEntry:
    model_id: str
    path: Path
    source: str  # "run" | "export"
    info: dict[str, Any]


class ModelRegistry:
    def __init__(self, runs: RunIndex, models_dir: Path) -> None:
        self.runs = runs
        self.models_dir = models_dir
        self._lock = threading.Lock()
        self._models: dict[str, ModelEntry] = {}
        self._actor_cache: dict[str, tuple[Any, dict]] = {}

    def refresh(self) -> None:
        found: dict[str, ModelEntry] = {}
        for run_id, run in self.runs.paths().items():
            actor = run / "actor.pt"
            manifest = read_json(run / "manifest.json")
            if not actor.is_file() or not isinstance(manifest, dict) or manifest.get("status") != "completed":
                continue  # only finished runs are offered
            found[run_id] = ModelEntry(run_id, actor, "run", scrub(self._run_info(run_id, run, manifest)))
        if self.models_dir.is_dir():
            for actor in sorted(self.models_dir.glob("*/actor.pt")):
                name = actor.parent.name
                if is_safe_id(name) and name not in found:
                    found[name] = ModelEntry(name, actor, "export", {"model_id": name, "source": "export"})
        with self._lock:
            self._models = found
            self._actor_cache = {k: v for k, v in self._actor_cache.items() if k in found}

    @staticmethod
    def _run_info(run_id: str, run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        test_agg = None
        for f in sorted((run / "eval").glob("test_*.json")) if (run / "eval").is_dir() else []:
            test = read_json(f)
            modes = test.get("modes") if isinstance(test, dict) else None
            if isinstance(modes, dict) and isinstance(modes.get("stochastic"), dict):
                # Primary (stochastic) held-out aggregate, passed through unmodified.
                test_agg = modes["stochastic"].get("aggregate")
                break
        return {
            "model_id": run_id,
            "source": "run",
            "run_id": run_id,
            "study": manifest.get("study"),
            "algo": manifest.get("algo"),
            "level": manifest.get("level"),
            "seed": manifest.get("seed"),
            "trained_env_steps": manifest.get("trained_env_steps"),
            "final_val_metrics": manifest.get("final_val_metrics"),
            "test_metrics": test_agg,
        }

    def list(self, algo: str | None = None, level: str | None = None) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            entries = list(self._models.values())
        out = [e.info for e in entries]
        if algo:
            out = [m for m in out if m.get("algo") == algo]
        if level:
            out = [m for m in out if m.get("level") == level]
        return out

    def get(self, model_id: str) -> ModelEntry:
        with self._lock:
            entry = self._models.get(model_id)
        if entry is None:
            self.refresh()
            with self._lock:
                entry = self._models.get(model_id)
        if entry is None:
            raise NotFoundError(f"model '{model_id}' not found")
        return entry

    def load_actor(self, model_id: str) -> tuple[Any, dict, dict[str, Any]]:
        """Return `(actor, meta, info)`. Actors are cached (read-only, eval mode)."""
        from marl.algorithms.common.policy import load_actor

        entry = self.get(model_id)
        with self._lock:
            cached = self._actor_cache.get(model_id)
        if cached is None:
            try:
                cached = load_actor(str(entry.path))
            except Exception as exc:  # corrupt / incompatible file
                raise ValidationFailed(f"model '{model_id}' could not be loaded ({type(exc).__name__})") from exc
            with self._lock:
                self._actor_cache[model_id] = cached
        return cached[0], cached[1], entry.info

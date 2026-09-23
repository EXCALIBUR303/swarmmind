"""Run directory + manifest lifecycle (§21.1-21.2).

Layout: `results/runs/<study>/<algo>/<level>/s<seed>-<timestamp>-<hash8>/`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from marl.utils.hashing import config_hash
from marl.utils.paths import RUNS_DIR
from marl.utils.versions import collect_versions


def make_run_dir(
    study: str,
    algo: str,
    level: str,
    seed: int,
    resolved_config: dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    root = base_dir or RUNS_DIR
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    h8 = config_hash(resolved_config)
    run_dir = root / study / algo / level / f"s{seed}-{timestamp}-{h8}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


class RunManifest:
    """Manifest read/write (§21.2). `status` moves running -> completed |
    failed | interrupted."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "manifest.json"
        self.data: dict[str, Any] = {}

    @classmethod
    def create(
        cls,
        run_dir: Path,
        study: str,
        algo: str,
        level: str,
        seed: int,
        env_config: dict[str, Any],
        algo_config: dict[str, Any],
        config_hash_value: str,
    ) -> RunManifest:
        m = cls(run_dir)
        m.data = {
            "run_id": run_dir.name,
            "study": study,
            "algo": algo,
            "level": level,
            "seed": seed,
            "env_config": env_config,
            "algo_config": algo_config,
            "total_env_steps": 0,
            "total_agent_steps": 0,
            "updates": 0,
            "start_time": datetime.now(UTC).isoformat(),
            "end_time": None,
            "wall_time_seconds": None,
            "versions": collect_versions(),
            "config_hash": config_hash_value,
            "checkpoints": [],
            "final_val_metrics": None,
            "status": "running",
        }
        m.save()
        return m

    @classmethod
    def load(cls, run_dir: Path) -> RunManifest:
        m = cls(run_dir)
        with m.path.open("r") as f:
            m.data = json.load(f)
        return m

    def save(self) -> None:
        """Atomic write: write to a temp file, then rename (§15.10)."""
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(self.data, f, indent=2, default=str)
        tmp.replace(self.path)

    def update_progress(self, env_steps: int, agent_steps: int, updates: int) -> None:
        self.data["total_env_steps"] = env_steps
        self.data["total_agent_steps"] = agent_steps
        self.data["updates"] = updates
        self.save()

    def add_checkpoint(self, checkpoint_info: dict[str, Any]) -> None:
        self.data["checkpoints"].append(checkpoint_info)
        self.save()

    def finish(self, status: str, final_val_metrics: dict[str, Any] | None = None) -> None:
        self.data["status"] = status
        self.data["end_time"] = datetime.now(UTC).isoformat()
        start = datetime.fromisoformat(self.data["start_time"])
        end = datetime.fromisoformat(self.data["end_time"])
        self.data["wall_time_seconds"] = (end - start).total_seconds()
        if final_val_metrics is not None:
            self.data["final_val_metrics"] = final_val_metrics
        self.save()

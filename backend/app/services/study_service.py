"""Reads aggregated study outputs (Phase 6-8 artifacts) from
`results/studies/<study>/` (§28.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.services.errors import NotFoundError, ValidationFailed
from backend.app.services.jsonio import read_json, scrub
from marl.utils.paths import UnsafePathError, is_safe_id, safe_join

HEATMAP_KINDS = {"visits": "visit_counts", "collisions": "collision_counts"}


class StudyService:
    def __init__(self, studies_dir: Path) -> None:
        self.studies_dir = studies_dir

    def _dir(self, study_id: str) -> Path:
        try:
            path = safe_join(self.studies_dir, study_id)
        except UnsafePathError as exc:
            raise NotFoundError(f"study '{study_id}' not found") from exc
        if not (path / "summary.json").is_file():
            raise NotFoundError(f"study '{study_id}' not found")
        return path

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.studies_dir.is_dir():
            return out
        for d in sorted(self.studies_dir.iterdir()):
            summary = read_json(d / "summary.json") if d.is_dir() and is_safe_id(d.name) else None
            if not isinstance(summary, dict):
                continue
            out.append(
                {
                    "study": d.name,
                    "smoke": summary.get("smoke"),
                    "generated": summary.get("generated"),
                    "levels": sorted((summary.get("levels") or {}).keys()),
                    "primary": summary.get("primary"),
                }
            )
        return out

    def get(self, study_id: str) -> dict[str, Any]:
        d = self._dir(study_id)
        manifest = read_json(d / "study_manifest.json")
        trimmed = None
        if isinstance(manifest, dict):
            trimmed = {k: manifest.get(k) for k in ("study", "created", "finished", "smoke", "deviations", "stages", "inputs_sha256") if k in manifest}
            trimmed["n_jobs"] = len(manifest.get("jobs") or [])
        analysis = d / "analysis"
        return scrub(
            {
                "summary": read_json(d / "summary.json"),
                "manifest": trimmed,
                "analysis_files": sorted(f.name for f in analysis.glob("*.json")) if analysis.is_dir() else [],
            }
        )

    def heatmap(self, study_id: str, level: str, algo: str, map_id: str, kind: str) -> dict[str, Any]:
        if kind not in HEATMAP_KINDS:
            raise ValidationFailed(f"kind must be one of {sorted(HEATMAP_KINDS)}")
        d = self._dir(study_id)
        for part in (level, algo, map_id):
            if not is_safe_id(part):
                raise ValidationFailed("invalid identifier")
        path = d / "analysis" / f"{level}_{algo}_{map_id}_heatmap.json"
        data = read_json(path)
        if not isinstance(data, dict):
            raise NotFoundError("no precomputed heatmap for that study/level/algo/map")
        return {
            "study": study_id,
            "level": level,
            "algo": algo,
            "map_id": data.get("map_id", map_id),
            "kind": kind,
            "n_episodes": data.get("n_episodes"),
            "obstacles": data.get("obstacles"),
            "counts": data.get(HEATMAP_KINDS[kind]),
        }

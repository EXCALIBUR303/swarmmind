"""Level configs and fixed evaluation maps (§28.2).

Held-out splits (test, structured) go through `marl`'s gatekeeper, which
writes the access log (§20.3). Validation maps are open."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.services.errors import NotFoundError, ValidationFailed
from marl.envs.core.config import EnvConfig, LevelRegistry
from marl.envs.core.maps_io import MapSpec
from marl.evaluation.eval_sets import eval_set_exists, load_eval_set
from marl.evaluation.final_eval import load_heldout_maps

MAP_SETS = ("validation", "test", "structured")


def map_to_dict(split: str, level: str, m: MapSpec) -> dict[str, Any]:
    return {
        "id": m.id,
        "set": split,
        "level": level,
        "hash": m.content_hash,
        "width": m.width,
        "height": m.height,
        "n_robots": len(m.robot_starts),
        "n_targets": len(m.targets),
        "terrain": m.terrain_rows,
        "robot_starts": [list(p) for p in m.robot_starts],
        "targets": [list(p) for p in m.targets],
    }


class MapService:
    def __init__(self, access_log: Path) -> None:
        self.access_log = access_log
        self.levels = LevelRegistry()

    def list_levels(self) -> list[dict[str, Any]]:
        return [{"level": name, "config": cfg.model_dump()} for name, cfg in self.levels.all().items()]

    def level_config(self, level: str) -> EnvConfig:
        try:
            return self.levels.get(level)
        except KeyError as exc:
            raise NotFoundError(f"unknown level '{level}'") from exc

    def load_set(self, split: str, level: str) -> list[MapSpec]:
        if split not in MAP_SETS:
            raise ValidationFailed(f"set must be one of {list(MAP_SETS)}")
        self.level_config(level)
        if not eval_set_exists(level, split):
            return []
        if split == "validation":
            return load_eval_set(level, split)
        return load_heldout_maps(level, split, {"who": "backend_api"}, None, self.access_log)

    def list_maps(self, split: str, level: str | None = None) -> list[dict[str, Any]]:
        levels = [level] if level else self.levels.available_levels()
        return [map_to_dict(split, lv, m) for lv in levels for m in self.load_set(split, lv)]

    def find_map(self, split: str, level: str, map_id: str) -> MapSpec:
        for m in self.load_set(split, level):
            if m.id == map_id:
                return m
        raise NotFoundError(f"map '{map_id}' not found in {split}/{level}")

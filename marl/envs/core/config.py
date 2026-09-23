"""Environment configuration schema (§6.6 Episode configuration, §6.7 Levels).

Phase 1 scope: the config schema and level registry only. No environment
dynamics (SwarmWorld, mapgen, etc.) are implemented yet — those are Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from marl.utils.config_loader import load_yaml
from marl.utils.paths import CONFIGS_DIR

CommMode = Literal["none", "shared_map", "limited_range"]
SpawnMode = Literal["clustered", "scattered"]


class SensingConfig(BaseModel):
    occlusion: bool = False


class RewardConfig(BaseModel):
    """Weights for the reward function (§13.2)."""

    explore_total: float = Field(1.0, ge=0.0, description="w_x total budget (w_x = explore_total/|G|)")
    discover: float = Field(0.25, ge=0.0)
    rescue: float = Field(1.0, ge=0.0)
    complete: float = Field(2.0, ge=0.0)
    time_cost: float = Field(0.01, ge=0.0)
    collision: float = Field(0.05, ge=0.0)
    team_share: float = Field(1.0, ge=0.0, le=1.0, description="β: fraction of positive reward shared")


class ObsConfig(BaseModel):
    agent_id_onehot: bool = Field(False, description="Ablation only; default excludes agent identity")


class EnvConfig(BaseModel):
    """Full episode/environment configuration (§6.6).

    A named level (L1-L4, §6.7) is simply a frozen instance of this model,
    loaded from configs/env/<Level>.yaml.
    """

    name: str
    width: int = Field(..., ge=6, le=20)
    height: int = Field(..., ge=6, le=20)
    n_robots: int = Field(..., ge=1, le=6)
    n_targets: int = Field(..., ge=1, le=5)
    obstacle_density: float = Field(..., ge=0.0, le=0.35)
    max_steps: int = Field(..., ge=1)
    sensor_radius: float = Field(..., ge=1.0)
    obs_radius: int = Field(5, ge=1)
    comm_mode: CommMode = "shared_map"
    comm_radius: float = Field(4.0, ge=0.0)
    spawn: SpawnMode = "clustered"
    target_min_dist: int = Field(3, ge=0)
    action_masking: bool = True
    sensing: SensingConfig = Field(default_factory=SensingConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    obs: ObsConfig = Field(default_factory=ObsConfig)

    @model_validator(mode="after")
    def _check_capacity_and_consistency(self) -> EnvConfig:
        n_cells = self.width * self.height
        n_needed = self.n_robots + self.n_targets
        # Loose capacity check; the real reachable-free-cell check happens
        # against the generated map in Phase 2 (mapgen validates strictly).
        max_occupiable = int(n_cells * (1.0 - self.obstacle_density))
        if n_needed > max_occupiable:
            raise ValueError(
                f"level '{self.name}': n_robots+n_targets={n_needed} exceeds the "
                f"expected occupiable capacity ({max_occupiable}) of a "
                f"{self.width}x{self.height} grid at density {self.obstacle_density}"
            )
        if self.obs_radius < self.sensor_radius:
            raise ValueError(
                f"level '{self.name}': obs_radius ({self.obs_radius}) must be >= "
                f"sensor_radius ({self.sensor_radius}) so the sensed disc fits "
                "inside the observation window"
            )
        if self.comm_mode == "limited_range" and self.comm_radius <= 0:
            raise ValueError(
                f"level '{self.name}': comm_mode='limited_range' requires comm_radius > 0"
            )
        return self


_LEVEL_FILES = {
    "L1": CONFIGS_DIR / "env" / "L1.yaml",
    "L2": CONFIGS_DIR / "env" / "L2.yaml",
    "L3": CONFIGS_DIR / "env" / "L3.yaml",
    "L4": CONFIGS_DIR / "env" / "L4.yaml",
}


class LevelRegistry:
    """Loads and caches the named level configs (L1-L4) from configs/env/."""

    def __init__(self, level_files: dict[str, Path] | None = None) -> None:
        self._level_files = level_files or _LEVEL_FILES
        self._cache: dict[str, EnvConfig] = {}

    def available_levels(self) -> list[str]:
        return sorted(self._level_files.keys())

    def get(self, level: str) -> EnvConfig:
        if level not in self._level_files:
            raise KeyError(
                f"unknown level '{level}'; available levels: {self.available_levels()}"
            )
        if level not in self._cache:
            raw = load_yaml(self._level_files[level])
            self._cache[level] = EnvConfig.model_validate(raw)
        return self._cache[level]

    def all(self) -> dict[str, EnvConfig]:
        return {lvl: self.get(lvl) for lvl in self.available_levels()}

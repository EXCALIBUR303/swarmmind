"""Study configuration (§32, Appendix A.3): `configs/experiments/<study>.yaml`.

A study is the cross product levels × algos × seeds at one env-step
budget, plus non-learning baselines and the final held-out evaluation.
IPPO and MAPPO resolve from `configs/algo/{ippo,mappo}.yaml` with the SAME
`algo_overrides` applied to both, so a study can never give the two
algorithms different hyperparameters (§16.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from marl.algorithms.common.config import AlgoConfig
from marl.utils.config_loader import apply_overrides, load_yaml
from marl.utils.paths import CONFIGS_DIR, is_safe_id

LearnedAlgo = Literal["ippo", "mappo"]
HeldOutSplit = Literal["test", "structured"]


class RandomBaselineConfig(BaseModel):
    # Random's unit of analysis: one evaluation sampling seed = one "seed".
    eval_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4], min_length=1)


class HeuristicBaselineConfig(BaseModel):
    reference_only: bool = True  # context line only; never a hypothesis arm (§14.2)


class BaselinesConfig(BaseModel):
    random: RandomBaselineConfig = Field(default_factory=RandomBaselineConfig)
    heuristic: HeuristicBaselineConfig | None = Field(default_factory=HeuristicBaselineConfig)


def _default_splits() -> list[HeldOutSplit]:
    return ["test", "structured"]


class FinalEvalConfig(BaseModel):
    map_sets: list[HeldOutSplit] = Field(default_factory=_default_splits, min_length=1)
    episodes_per_map: int = Field(2, ge=1)  # evaluation sampling seeds 0..k-1 (§22.1)
    stochastic: bool = True  # primary mode; must stay True (§22.3)
    also_greedy: bool = True  # secondary mode, reported separately
    checkpoint: Literal["final"] = "final"  # primary checkpoint (§22.1)

    @field_validator("stochastic")
    @classmethod
    def _stochastic_primary(cls, v: bool) -> bool:
        if not v:
            raise ValueError("stochastic evaluation is the pre-registered primary mode (§22.3)")
        return v

    @property
    def eval_seeds(self) -> list[int]:
        return list(range(self.episodes_per_map))

    @property
    def modes(self) -> list[str]:
        return ["stochastic", "greedy"] if self.also_greedy else ["stochastic"]


class EvolutionReplayConfig(BaseModel):
    fractions: list[float] = Field(default_factory=lambda: [0.0, 0.1, 0.5, 1.0])
    n_maps: int = Field(5, ge=1)
    split: Literal["validation", "test", "structured"] = "structured"
    seeds: list[int] = Field(default_factory=lambda: [0])  # which training seeds to record


class AnalysisConfig(BaseModel):
    """Pre-registered analysis choices. Defaults are the Phase 0 spec (§4, §22.2)."""

    # H2's decision metric. Spec default: success. "censored_steps" (failures
    # counted as max_steps; lower is better) is the alternative for levels
    # where success saturates for every method.
    h2_primary_metric: Literal["success_rate", "censored_steps"] = "success_rate"
    # Validation-success AUC window in env steps; None = the whole budget.
    auc_window_env_steps: int | None = Field(None, gt=0)


class StudyConfig(BaseModel):
    study: str
    smoke: bool = False  # SMOKE studies are infrastructure dry runs: never evidence
    description: str = ""
    levels: list[str] = Field(min_length=1)
    algos: list[LearnedAlgo] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    total_env_steps: int = Field(gt=0)
    algo_overrides: dict[str, Any] = Field(default_factory=dict)  # dotted keys -> value, applied to EVERY algo
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    final_eval: FinalEvalConfig = Field(default_factory=FinalEvalConfig)
    evolution_replays: EvolutionReplayConfig | None = Field(default_factory=EvolutionReplayConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @field_validator("study")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not is_safe_id(v):
            raise ValueError(f"study name {v!r} must match [A-Za-z0-9._-]{{1,128}}")
        return v

    @model_validator(mode="after")
    def _unique(self) -> StudyConfig:
        for name in ("levels", "algos", "seeds"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate entries in {name}: {values}")
        if any(k.startswith("critic") for k in self.algo_overrides):
            raise ValueError("algo_overrides may not touch critic.* — that is the controlled variable (§16.3)")
        return self

    def resolve_algo(self, algo: str) -> AlgoConfig:
        raw = load_yaml(CONFIGS_DIR / "algo" / f"{algo}.yaml")
        overrides = [f"{k}={v}" for k, v in self.algo_overrides.items()]
        return AlgoConfig.model_validate(apply_overrides(raw, overrides) if overrides else raw)


def study_config_path(study: str) -> Path:
    return CONFIGS_DIR / "experiments" / f"{study}.yaml"


def load_study_config(path_or_name: str | Path) -> StudyConfig:
    path = Path(path_or_name)
    if not path.suffix:
        path = study_config_path(str(path_or_name))
    return StudyConfig.model_validate(load_yaml(path))

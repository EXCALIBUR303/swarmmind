"""Request/response models for live sessions and the sim WebSocket (§28.2-28.3).

Frames reuse `marl.replay.schema.ReplayFrame`, so the live stream and the
recorded replays share one frame type."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marl.envs.core.config import EnvConfig
from marl.replay.schema import ReplayFrame, ReplayMapInfo, ReplayPolicyInfo

ID_PATTERN = r"^[A-Za-z0-9._-]{1,128}$"


class ProceduralMap(BaseModel):
    kind: Literal["procedural"] = "procedural"


class FixedMap(BaseModel):
    kind: Literal["fixed"] = "fixed"
    set: Literal["validation", "test", "structured"]
    id: str = Field(pattern=ID_PATTERN)


MapChoice = Annotated[ProceduralMap | FixedMap, Field(discriminator="kind")]


class PolicyChoice(BaseModel):
    kind: Literal["random", "heuristic", "ippo", "mappo"]
    model_id: str | None = Field(default=None, pattern=ID_PATTERN)

    @model_validator(mode="after")
    def _model_required_iff_learned(self) -> PolicyChoice:
        learned = self.kind in ("ippo", "mappo")
        if learned and not self.model_id:
            raise ValueError(f"policy kind '{self.kind}' requires model_id")
        if not learned and self.model_id:
            raise ValueError(f"policy kind '{self.kind}' does not take a model_id")
        return self


class SessionCreate(BaseModel):
    """Exactly one of `level` / `env_config`. `seed` generates the map when
    it is procedural and ALWAYS seeds the policy's sampling stream, exactly
    like `eval_seed` in the batch evaluator (§28.4)."""

    level: str | None = Field(default=None, pattern=ID_PATTERN)
    env_config: EnvConfig | None = None
    map: MapChoice = Field(default_factory=ProceduralMap)
    policy: PolicyChoice
    stochastic: bool = True
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _level_xor_config(self) -> SessionCreate:
        if (self.level is None) == (self.env_config is None):
            raise ValueError("provide exactly one of 'level' or 'env_config'")
        return self


class SessionReset(BaseModel):
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    map: MapChoice | None = None


class StepRequest(BaseModel):
    n: int = Field(default=1, ge=1)  # upper bound enforced against settings in the router


class LiveMetrics(BaseModel):
    t: int
    coverage: float
    n_discovered: int
    n_rescued: int
    n_collisions: int
    return_mean: float
    success: bool
    done: bool


class SessionInitial(BaseModel):
    pos: list[list[int]]
    known_team: list[list[int]]
    # Targets already sensed at reset; no frame's `discovered` contains them (Phase 15 fix).
    discovered: list[int] = Field(default_factory=list)


class SessionSnapshot(BaseModel):
    session_id: str
    step_index: int
    done: bool
    stochastic: bool
    seed: int
    env: dict[str, Any]
    map: ReplayMapInfo
    policy: ReplayPolicyInfo
    initial: SessionInitial
    current_pos: list[list[int]]
    current_known_team: list[list[int]]
    target_status: list[int]
    last_frame: ReplayFrame | None
    metrics: LiveMetrics
    final_metrics: dict[str, Any] | None = None


class StepResponse(BaseModel):
    session_id: str
    frames: list[ReplayFrame]
    step_index: int
    done: bool
    metrics: LiveMetrics
    final_metrics: dict[str, Any] | None = None


# --- WebSocket protocol (documented + exported by scripts/export_schemas.py) ---


class WsCommand(BaseModel):
    model_config = ConfigDict(strict=True)  # no bool->number or str->number coercion on the wire

    cmd: Literal["play", "pause", "step", "reset", "speed"]
    value: float | int | dict[str, Any] | None = None


class WsFrameMessage(BaseModel):
    type: Literal["frame"] = "frame"
    frame: ReplayFrame
    step_index: int
    metrics: LiveMetrics


class WsStateMessage(BaseModel):
    type: Literal["state"] = "state"
    playing: bool
    speed: float
    snapshot: SessionSnapshot


class WsEpisodeEnd(BaseModel):
    type: Literal["episode_end"] = "episode_end"
    metrics: dict[str, Any]
    success: bool
    steps: int


class WsError(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str

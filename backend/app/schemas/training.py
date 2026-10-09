"""Training-job and training-stream models (§28.2, §28.5, §37).

The submitted parameters are bounded here AND re-validated by the same
`AlgoConfig` schema the CLI uses (see `job_runner`)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# key -> (type, lo, hi). Only these overrides can be submitted from the UI;
# the key and value are never interpolated into anything but an argv element.
ALLOWED_OVERRIDES: dict[str, tuple[type, float, float]] = {
    "ppo.lr_actor": (float, 1e-6, 1e-2),
    "ppo.lr_critic": (float, 1e-6, 1e-2),
    "ppo.entropy_coef": (float, 0.0, 0.1),
    "ppo.clip": (float, 0.05, 0.5),
    "ppo.gamma": (float, 0.9, 0.9999),
    "ppo.gae_lambda": (float, 0.8, 1.0),
    "ppo.epochs": (int, 1, 20),
    "ppo.n_envs": (int, 1, 32),
    "ppo.rollout_len": (int, 16, 512),
    "ppo.minibatches": (int, 1, 16),
    "eval.interval_updates": (int, 1, 1000),
}

MIN_UI_ENV_STEPS = 2048


class TrainRequest(BaseModel):
    level: Literal["L1", "L2", "L3", "L4"]
    algo: Literal["ippo", "mappo"]
    seed: int = Field(ge=0, le=2**31 - 1)
    total_env_steps: int = Field(ge=MIN_UI_ENV_STEPS, le=10_000_000)
    overrides: dict[str, float | int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_overrides(self) -> TrainRequest:
        for key, value in self.overrides.items():
            if key not in ALLOWED_OVERRIDES:
                raise ValueError(f"override '{key}' is not allowed (allowed: {sorted(ALLOWED_OVERRIDES)})")
            typ, lo, hi = ALLOWED_OVERRIDES[key]
            if isinstance(value, bool) or (typ is int and int(value) != value):
                raise ValueError(f"override '{key}' must be {typ.__name__}")
            if not lo <= value <= hi:
                raise ValueError(f"override '{key}' must be within [{lo}, {hi}]")
        return self


class JobStatus(BaseModel):
    job_id: str
    status: Literal["starting", "running", "completed", "failed", "cancelled"]
    run_id: str | None = None
    request: TrainRequest
    created_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    run_status: str | None = None
    progress: dict[str, Any] | None = None
    log_tail: list[str] | None = None  # last lines of the job's output; only for failed/cancelled jobs


def train_schema(max_env_steps: int) -> dict[str, Any]:
    """The form's bounds, from the same constants the validator uses (never duplicated by hand)."""
    return {
        "levels": ["L1", "L2", "L3", "L4"],
        "algos": ["ippo", "mappo"],
        "seed": {"min": 0, "max": 2**31 - 1},
        "total_env_steps": {"min": MIN_UI_ENV_STEPS, "max": max_env_steps},
        "overrides": {k: {"type": t.__name__, "min": lo, "max": hi} for k, (t, lo, hi) in ALLOWED_OVERRIDES.items()},
    }

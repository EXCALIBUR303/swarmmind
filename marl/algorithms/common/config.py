"""Algorithm configuration schema, matching configs/algo/{ippo,mappo}.yaml (§15.8).

IPPO and MAPPO share this exact schema; the ONLY field that should ever
differ between their two config files is `critic.input`
("local" vs "agent_specific_global") — everything else (network sizes,
PPO hyperparameters, checkpoint schedule) must be identical for the
algorithm comparison to isolate that one variable (§16.3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CriticInput = Literal["local", "agent_specific_global"]
LrSchedule = Literal["constant", "linear"]
Device = Literal["auto", "cpu", "cuda", "mps"]


class CriticConfig(BaseModel):
    input: CriticInput = "local"


class ActorConfig(BaseModel):
    encoder: str = "local_cnn_v1"
    hidden: int = Field(256, gt=0)


class PPOConfig(BaseModel):
    n_envs: int = Field(16, gt=0)
    rollout_len: int = Field(128, gt=0)
    epochs: int = Field(5, gt=0)
    minibatches: int = Field(4, gt=0)
    clip: float = Field(0.2, gt=0.0)
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(0.95, ge=0.0, le=1.0)
    lr_actor: float = Field(3.0e-4, gt=0.0)
    lr_critic: float = Field(3.0e-4, gt=0.0)
    adam_eps: float = Field(1.0e-5, gt=0.0)
    entropy_coef: float = Field(0.01, ge=0.0)
    max_grad_norm: float = Field(0.5, gt=0.0)
    value_norm: bool = True
    value_clip: bool = False
    target_kl: float | None = None
    lr_schedule: LrSchedule = "constant"


class EvalConfig(BaseModel):
    interval_updates: int = Field(20, gt=0)
    n_val_maps: int = Field(50, gt=0)
    stochastic: bool = True


class CheckpointConfig(BaseModel):
    fractions: list[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )


class AlgoConfig(BaseModel):
    name: str
    critic: CriticConfig = Field(default_factory=CriticConfig)
    actor: ActorConfig = Field(default_factory=ActorConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    device: Device = "auto"
    deterministic: bool = False

    @property
    def is_mappo(self) -> bool:
        return self.critic.input == "agent_specific_global"

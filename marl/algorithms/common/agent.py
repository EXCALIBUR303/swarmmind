"""PPOAgent: the algorithm-agnostic actor+critic+optimizer bundle the
trainer drives (§15.9). IPPO and MAPPO each subclass this and implement
ONLY `_critic_forward` — everything else (rollout action selection,
evaluate-for-update) is shared, which is what makes the training loop
itself algorithm-agnostic and keeps the two algorithms' code paths
identical apart from that one method (§16.3 parity requirement).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam

from marl.algorithms.common.distributions import MaskedCategorical
from marl.algorithms.common.value_norm import ValueNorm
from marl.networks.actor import ActorPolicy


@dataclass
class CriticInputs:
    """Everything a critic MIGHT need. IPPO's `_critic_forward` reads only
    `local_grid`/`local_vec`; MAPPO's (Phase 5) also reads
    `state_grid`/`state_vec`. Passing both unconditionally keeps the
    trainer's call sites identical for either algorithm."""

    local_grid: torch.Tensor
    local_vec: torch.Tensor
    state_grid: torch.Tensor | None = None
    state_vec: torch.Tensor | None = None


class PPOAgent(ABC):
    # Whether the trainer must collect, store and pass agent-specific global
    # state (§9.3). False for IPPO; True for MAPPO. The trainer's data path
    # is the ONLY thing this switches — it must never change the actor, the
    # RNG streams, or the order of operations (IPPO/MAPPO parity, §16.3).
    uses_global_state: bool = False

    def __init__(
        self,
        actor: ActorPolicy,
        critic: nn.Module,
        lr_actor: float,
        lr_critic: float,
        adam_eps: float,
        value_norm: bool = True,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.opt_actor = Adam(actor.parameters(), lr=lr_actor, eps=adam_eps)
        self.opt_critic = Adam(critic.parameters(), lr=lr_critic, eps=adam_eps)
        self.value_norm = ValueNorm() if value_norm else None

    @abstractmethod
    def _critic_forward(self, inputs: CriticInputs) -> torch.Tensor:
        """The critic head's raw output, shape [B]. When ValueNorm is on
        this is in NORMALIZED space (the head is trained against
        normalized targets, §16.4)."""
        raise NotImplementedError

    def critic_value_denormalized(self, inputs: CriticInputs) -> torch.Tensor:
        """Value in REWARD scale — what GAE needs (§15.6, §16.4: GAE uses
        denormalized values)."""
        normalized = self._critic_forward(inputs)
        if self.value_norm is None:
            return normalized
        return self.value_norm.denormalize(normalized)

    @torch.no_grad()
    def collect(
        self,
        grid: torch.Tensor,
        vec: torch.Tensor,
        mask: torch.Tensor,
        inputs: CriticInputs,
        generator: torch.Generator | None,
        stochastic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout-collection step: sample actions + get the DENORMALIZED
        value estimate for this observation (§15.9 loop). Returns
        (actions, log_probs, value_denormalized)."""
        actions, log_probs = self.actor.act(grid, vec, mask, generator, stochastic)
        value = self.critic_value_denormalized(inputs)
        return actions, log_probs, value

    def evaluate(
        self, grid: torch.Tensor, vec: torch.Tensor, mask: torch.Tensor, inputs: CriticInputs
    ) -> tuple[MaskedCategorical, torch.Tensor]:
        """Update-time forward pass WITH gradients: returns (dist,
        normalized_value) — the critic loss compares this normalized value
        against normalized targets (§16.4)."""
        dist = self.actor(grid, vec, mask)
        normalized_value = self._critic_forward(inputs)
        return dist, normalized_value

    def parameters_actor(self):
        return self.actor.parameters()

    def parameters_critic(self):
        return self.critic.parameters()

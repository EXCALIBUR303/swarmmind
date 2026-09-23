"""ActorPolicy: the decentralized actor network (§15.2).

CRITICAL: this module must NEVER import anything from `global_state` or
`marl.networks.critics` — enforced by `tests/test_import_boundaries.py`
(§12.4 test_actor_import_boundary). The actor's forward signature takes
ONLY (grid, vec, mask): no other tensor input exists, by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from marl.algorithms.common.distributions import MaskedCategorical
from marl.envs.core.types import N_ACTIONS
from marl.networks.encoders import LOCAL_HIDDEN, LocalEncoder
from marl.networks.init import GAIN_POLICY_HEAD, init_head_


class ActorPolicy(nn.Module):
    """Shared, parameter-shared decentralized actor (§17).

    forward(grid, vec, mask) -> MaskedCategorical. This exact signature
    (§12.4 test_actor_forward_signature) is what makes the actor's
    information boundary auditable: there is no other tensor input it
    could possibly read privileged information through.
    """

    def __init__(self, grid_size: int = 11, hidden: int = LOCAL_HIDDEN) -> None:
        super().__init__()
        self.encoder = LocalEncoder(grid_size=grid_size)
        self.policy_head = nn.Linear(hidden, N_ACTIONS)
        init_head_(self.policy_head, GAIN_POLICY_HEAD)

    def forward(self, grid: torch.Tensor, vec: torch.Tensor, mask: torch.Tensor) -> MaskedCategorical:
        h = self.encoder(grid, vec)
        logits = self.policy_head(h)
        return MaskedCategorical(logits, mask)

    @torch.no_grad()
    def act(
        self,
        grid: torch.Tensor,
        vec: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
        stochastic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (or greedily select) actions and return (actions, log_probs).

        Used at rollout-collection time and at evaluation time — NOT
        during the PPO update itself, which instead calls `forward()`
        directly to recompute log-probs/entropy with gradients enabled.
        """
        dist = self.forward(grid, vec, mask)
        actions = dist.sample(generator) if stochastic else dist.mode()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

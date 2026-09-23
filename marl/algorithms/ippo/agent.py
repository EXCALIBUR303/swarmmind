"""IPPOAgent: parameter-shared actor + parameter-shared LOCAL critic (§15.1).

The critic sees ONLY what the actor sees — this is the entire definition
of IPPO relative to MAPPO (§16.3): everything else (network sizes, PPO
hyperparameters, training loop) is identical.
"""

from __future__ import annotations

import torch

from marl.algorithms.common.agent import CriticInputs, PPOAgent
from marl.algorithms.common.config import AlgoConfig
from marl.networks.actor import ActorPolicy
from marl.networks.critics import LocalCritic


class IPPOAgent(PPOAgent):
    def __init__(self, cfg: AlgoConfig, grid_size: int = 11) -> None:
        if cfg.is_mappo:
            raise ValueError("IPPOAgent requires critic.input == 'local'; got 'agent_specific_global'")
        actor = ActorPolicy(grid_size=grid_size, hidden=cfg.actor.hidden)
        critic = LocalCritic(grid_size=grid_size, hidden=cfg.actor.hidden)
        super().__init__(
            actor=actor,
            critic=critic,
            lr_actor=cfg.ppo.lr_actor,
            lr_critic=cfg.ppo.lr_critic,
            adam_eps=cfg.ppo.adam_eps,
            value_norm=cfg.ppo.value_norm,
        )

    def _critic_forward(self, inputs: CriticInputs) -> torch.Tensor:
        return self.critic(inputs.local_grid, inputs.local_vec)

"""MAPPOAgent: parameter-shared decentralized actor + parameter-shared
CENTRALIZED, agent-specific critic (§16).

Relative to IPPO, exactly one thing changes: the critic's input. The actor
is the identical `ActorPolicy` class, built first under the same seed, so
IPPO and MAPPO start from bitwise-identical actors (§16.3). The critic sees
the agent-specific global state (§9.3):

    x_i = (7 shared privileged channels ⊕ agent i's position ⊕ agent i's
           belief mask, 4-dim global vector, agent i's own local obs o_i)

Including o_i guarantees the critic conditions on a superset of what the
actor sees. Nothing here is ever reachable from the actor: the critic's
global inputs arrive only via `CriticInputs.state_grid/state_vec`, which
the actor's forward signature cannot accept (§12.4).
"""

from __future__ import annotations

import torch

from marl.algorithms.common.agent import CriticInputs, PPOAgent
from marl.algorithms.common.config import AlgoConfig
from marl.envs.core.global_state import N_GLOBAL_CHANNELS, N_GLOBAL_VEC_FEATURES
from marl.networks.actor import ActorPolicy
from marl.networks.critics import CentralCritic


class MAPPOAgent(PPOAgent):
    uses_global_state = True

    def __init__(self, cfg: AlgoConfig, grid_size: int = 11) -> None:
        if not cfg.is_mappo:
            raise ValueError("MAPPOAgent requires critic.input == 'agent_specific_global'; got 'local'")
        # Actor FIRST, exactly as IPPOAgent does, so a given init seed yields
        # the same actor weights for both algorithms.
        actor = ActorPolicy(grid_size=grid_size, hidden=cfg.actor.hidden)
        critic = CentralCritic(
            global_channels=N_GLOBAL_CHANNELS,
            grid_size=grid_size,
            hidden=cfg.actor.hidden,
            global_vec_dim=N_GLOBAL_VEC_FEATURES,
        )
        super().__init__(
            actor=actor,
            critic=critic,
            lr_actor=cfg.ppo.lr_actor,
            lr_critic=cfg.ppo.lr_critic,
            adam_eps=cfg.ppo.adam_eps,
            value_norm=cfg.ppo.value_norm,
        )

    def _critic_forward(self, inputs: CriticInputs) -> torch.Tensor:
        if inputs.state_grid is None or inputs.state_vec is None:
            raise ValueError("MAPPOAgent's centralized critic requires state_grid and state_vec")
        return self.critic(inputs.state_grid, inputs.state_vec, inputs.local_grid, inputs.local_vec)

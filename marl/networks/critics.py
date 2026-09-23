"""Critic networks (§15.2, §16.2): LocalCritic (IPPO) and CentralCritic (MAPPO).

Both are separate network instances from the actor's `LocalEncoder` (no
weight sharing between actor and critic, per §15.2) and from each other.
`CentralCritic` is defined here now but first constructed in Phase 5.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from marl.networks.encoders import GLOBAL_HIDDEN, LOCAL_HIDDEN, GlobalEncoder, LocalEncoder
from marl.networks.init import GAIN_VALUE_HEAD, init_head_, init_hidden_


class LocalCritic(nn.Module):
    """IPPO's critic: sees only the SAME local observation the actor sees
    (§15.1, §15.2). A separate `LocalEncoder` instance — no weight sharing
    with the actor."""

    def __init__(self, grid_size: int = 11, hidden: int = LOCAL_HIDDEN) -> None:
        super().__init__()
        self.encoder = LocalEncoder(grid_size=grid_size)
        self.value_head = nn.Linear(hidden, 1)
        init_head_(self.value_head, GAIN_VALUE_HEAD)

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        h = self.encoder(grid, vec)
        return self.value_head(h).squeeze(-1)


class CentralCritic(nn.Module):
    """MAPPO's centralized, agent-specific critic (§16.2). First used in
    Phase 5. Fuses a `GlobalEncoder` (shared+agent-specific privileged
    channels) with a SEPARATE `LocalEncoder'` (the actor's own local
    observation, §9.3) plus the 4-dim global vector.
    """

    def __init__(
        self,
        global_channels: int,
        grid_size: int = 11,
        hidden: int = LOCAL_HIDDEN,
        global_vec_dim: int = 4,
    ) -> None:
        super().__init__()
        self.global_encoder = GlobalEncoder(n_channels=global_channels)
        self.local_encoder = LocalEncoder(grid_size=grid_size)
        self.fuse = nn.Sequential(
            nn.Linear(GLOBAL_HIDDEN + hidden + global_vec_dim, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        init_head_(self.value_head, GAIN_VALUE_HEAD)
        init_hidden_(self.fuse)

    def forward(
        self,
        global_grid: torch.Tensor,
        global_vec: torch.Tensor,
        local_grid: torch.Tensor,
        local_vec: torch.Tensor,
    ) -> torch.Tensor:
        g = self.global_encoder(global_grid)
        loc = self.local_encoder(local_grid, local_vec)
        fused = self.fuse(torch.cat([g, loc, global_vec], dim=-1))
        return self.value_head(fused).squeeze(-1)

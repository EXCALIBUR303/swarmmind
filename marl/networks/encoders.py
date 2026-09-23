"""LocalEncoder and GlobalEncoder (§15.2, §16.2).

`LocalEncoder` is used, as SEPARATE instances (no weight sharing), for
both the actor and the IPPO/MAPPO local-critic pathway. `GlobalEncoder`
is used only by the MAPPO centralized critic (Phase 5) — it lives here
now so both encoders' shapes are defined in one place, but nothing in
Phase 4 constructs a `GlobalEncoder`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from marl.envs.core.observations import N_OBS_CHANNELS, N_VEC_FEATURES
from marl.networks.init import init_hidden_

LOCAL_HIDDEN = 256
GLOBAL_HIDDEN = 256


class LocalEncoder(nn.Module):
    """Encodes one robot's (grid[8,K,K], vec[10]) into a 256-dim embedding.

    K defaults to 11 (obs_radius=5, §8), which is what the spec's
    Conv/Flatten sizes (2304 = 64*6*6) are derived from. A different K
    still works (the conv stack is fully convolutional up to the flatten),
    but the flatten size is computed dynamically here rather than
    hardcoded, so the network adapts if `obs_radius` ever changes.
    """

    def __init__(self, grid_size: int = 11, n_channels: int = N_OBS_CHANNELS, n_vec: int = N_VEC_FEATURES) -> None:
        super().__init__()
        self.grid_size = grid_size

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_size = self._conv_output_size(n_channels, grid_size)
        self.grid_fc = nn.Sequential(nn.Linear(conv_out_size, LOCAL_HIDDEN), nn.ReLU())

        self.vec_fc = nn.Sequential(nn.Linear(n_vec, 64), nn.ReLU())

        self.fuse = nn.Sequential(nn.Linear(LOCAL_HIDDEN + 64, LOCAL_HIDDEN), nn.ReLU())

        init_hidden_(self)

    def _conv_output_size(self, n_channels: int, grid_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, grid_size, grid_size)
            out = self.conv(dummy)
            return int(out.numel())

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """grid: float[B,8,K,K], vec: float[B,10] -> h: float[B,256]."""
        b = grid.shape[0]
        conv_out = self.conv(grid).reshape(b, -1)
        grid_h = self.grid_fc(conv_out)
        vec_h = self.vec_fc(vec)
        return self.fuse(torch.cat([grid_h, vec_h], dim=-1))


class GlobalEncoder(nn.Module):
    """Encodes the shared global grid + vec into a 256-dim embedding (§16.2).

    Uses `AdaptiveAvgPool2d` so it accepts any map size H,W (§9.4) — this
    is the MAPPO centralized critic's encoder, first used in Phase 5.
    """

    def __init__(self, n_channels: int, n_vec: int = 4) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, GLOBAL_HIDDEN), nn.ReLU())
        init_hidden_(self)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        b = grid.shape[0]
        conv_out = self.conv(grid).reshape(b, -1)
        return self.fc(conv_out)

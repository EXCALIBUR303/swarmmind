"""Running value normalization (§16.4), used by BOTH IPPO and MAPPO so the
comparison stays fair (§16.3).

A running mean/std of returns via a beta-debiased EMA (matching the MAPPO
reference implementation's PopArt-free ValueNorm). The critic predicts
NORMALIZED values; GAE itself is computed on DENORMALIZED values.

Update order each PPO update (§16.4): `update(returns)` first, THEN
`normalize(returns)` to produce the regression targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    running_mean: torch.Tensor
    running_mean_sq: torch.Tensor
    debiasing_term: torch.Tensor

    def __init__(self, beta: float = 0.99999, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_mean_sq", torch.zeros(1))
        self.register_buffer("debiasing_term", torch.zeros(1))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1)
        batch_mean = values.mean()
        batch_mean_sq = (values**2).mean()

        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

    def _debiased_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        debiasing = torch.clamp(self.debiasing_term, min=self.epsilon)
        mean = self.running_mean / debiasing
        mean_sq = self.running_mean_sq / debiasing
        var = torch.clamp(mean_sq - mean**2, min=1e-2)
        return mean, var

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, var = self._debiased_mean_var()
        return (values - mean) / torch.sqrt(var)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, var = self._debiased_mean_var()
        return values * torch.sqrt(var) + mean

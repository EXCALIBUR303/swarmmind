"""The PPO clipped objective, value loss, and minibatch update loop (§15.7).

Diagnostics returned every update: policy loss, value loss, entropy,
approx KL, clip fraction, explained variance, gradient norms — all logged
by the trainer (§15.7 "Diagnostics logged per update").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from marl.algorithms.common.distributions import MaskedCategorical


@dataclass
class PPOLossOutputs:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (policy_loss, approx_kl, clip_fraction)."""
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > clip).float().mean()

    return policy_loss, approx_kl, clip_fraction


def value_loss_fn(
    new_values_normalized: torch.Tensor,
    target_returns_normalized: torch.Tensor,
) -> torch.Tensor:
    """0.5 * MSE, on NORMALIZED values/targets (§15.7). Value clipping is
    OFF by default (§15.8) — not implemented here to match that default;
    add if a future ablation needs it."""
    return 0.5 * ((new_values_normalized - target_returns_normalized) ** 2).mean()


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """1 - Var[returns - values] / Var[returns]. Diagnostic only."""
    var_returns = torch.var(returns)
    if var_returns.item() < 1e-8:
        return float("nan")
    return float(1.0 - torch.var(returns - values) / var_returns)


def compute_update_losses(
    dist: MaskedCategorical,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    new_values_normalized: torch.Tensor,
    target_returns_normalized: torch.Tensor,
    entropy_coef: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, PPOLossOutputs]:
    """One minibatch's actor loss and critic loss, plus diagnostics.

    Returns (actor_loss, critic_loss, diagnostics). The caller is
    responsible for separate optimizer steps on the actor and critic
    (§15.2: separate networks, separate optimizers).
    """
    new_logp = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    policy_loss, approx_kl, clip_fraction = ppo_policy_loss(new_logp, old_logp, advantages, clip)
    actor_loss = policy_loss - entropy_coef * entropy

    critic_loss = value_loss_fn(new_values_normalized, target_returns_normalized)

    diagnostics = PPOLossOutputs(
        policy_loss=policy_loss.detach(),
        value_loss=critic_loss.detach(),
        entropy=entropy.detach(),
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )
    return actor_loss, critic_loss, diagnostics


def clip_grad_norm(module: nn.Module, max_norm: float) -> float:
    return float(nn.utils.clip_grad_norm_(module.parameters(), max_norm))

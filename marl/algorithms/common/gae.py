"""Generalized Advantage Estimation with truncation folding (§15.5, §15.6).

This function assumes truncation bootstrapping has ALREADY been folded
into `rewards` by the caller (the trainer, when collecting a rollout):
for any env step where `truncated=True`, the caller must have already set
`rewards[t] += gamma * v_final` using the environment's TRUE final
observation (not the post-auto-reset one) — see
`marl.envs.vector.SyncVectorSwarm`'s `FinalInfo`. Once folded, every
`done_t = terminated_t or truncated_t` step is treated identically here:
no further bootstrap, full advantage decay reset.

All operations are on `[T, E, N]` tensors (time, env, agent) — GAE MUST be
computed on this un-flattened shape, since flattening first would let an
env's advantage leak into an adjacent env/agent's recursion (§16.8 item 2).
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,  # [T, E, N]
    values: torch.Tensor,  # [T, E, N], DENORMALIZED
    dones: torch.Tensor,  # [T, E], bool — terminated OR truncated
    last_value: torch.Tensor,  # [E, N], DENORMALIZED value at rollout end
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (advantages[T,E,N], returns[T,E,N]), both DENORMALIZED scale."""
    t_len, n_envs, n_agents = rewards.shape
    if values.shape != rewards.shape:
        raise ValueError(f"values shape {values.shape} must match rewards shape {rewards.shape}")
    if dones.shape != (t_len, n_envs):
        raise ValueError(f"dones shape {dones.shape} must be ({t_len}, {n_envs})")
    if last_value.shape != (n_envs, n_agents):
        raise ValueError(f"last_value shape {last_value.shape} must be ({n_envs}, {n_agents})")

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(n_envs, n_agents, dtype=rewards.dtype, device=rewards.device)

    not_done = (~dones).to(rewards.dtype).unsqueeze(-1)  # [T, E, 1], broadcasts over N

    for t in reversed(range(t_len)):
        next_value = last_value if t == t_len - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * not_done[t] - values[t]
        last_gae = delta + gamma * gae_lambda * not_done[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, active: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages over the WHOLE batch of active samples, once
    per update, BEFORE minibatching (§15.7). `active` is a same-shape bool
    mask (all-True in Phase 4/5; exists for future variable-team-size
    support, §16.6) — inactive samples are excluded from the mean/std."""
    flat = advantages.reshape(-1) if active is None else advantages[active]
    mean = flat.mean()
    std = flat.std(unbiased=False)
    return (advantages - mean) / (std + eps)

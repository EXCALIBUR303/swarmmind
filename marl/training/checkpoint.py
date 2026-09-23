"""Checkpoint save/load/resume, atomic writes, actor export (§15.10).

Full training checkpoints (`step_<n>.pt`, `latest.pt`) contain everything
needed to resume bitwise-identically: actor+critic state, both optimizer
states, ValueNorm stats, RNG states, step counters, and the resolved
config. Actor-only EXPORTS (`models/<name>/actor.pt`) contain ONLY the
actor's weights — loadable with `weights_only=True` and usable for
inference without ever importing critic code (§12.3).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl.algorithms.common.agent import PPOAgent

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    agent: PPOAgent,
    global_env_steps: int,
    update_idx: int,
    resolved_config: dict[str, Any],
    config_hash: str,
    global_agent_steps: int | None = None,
    extra_rng_state: dict[str, Any] | None = None,
) -> None:
    """Atomic write: write to a temp file, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "opt_actor_state_dict": agent.opt_actor.state_dict(),
        "opt_critic_state_dict": agent.opt_critic.state_dict(),
        "value_norm_state_dict": agent.value_norm.state_dict() if agent.value_norm is not None else None,
        "global_env_steps": global_env_steps,
        "global_agent_steps": global_agent_steps,
        "update_idx": update_idx,
        "resolved_config": resolved_config,
        "config_hash": config_hash,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            **(extra_rng_state or {}),
        },
    }

    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_agent(agent: PPOAgent, checkpoint: dict[str, Any]) -> None:
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.opt_actor.load_state_dict(checkpoint["opt_actor_state_dict"])
    agent.opt_critic.load_state_dict(checkpoint["opt_critic_state_dict"])
    if agent.value_norm is not None and checkpoint.get("value_norm_state_dict") is not None:
        agent.value_norm.load_state_dict(checkpoint["value_norm_state_dict"])


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def export_actor(
    path: str | Path,
    agent: PPOAgent,
    meta: dict[str, Any],
) -> None:
    """Export ONLY the actor's weights, for inference (§12.3, §15.10).
    Loadable with `weights_only=True`; never imports critic code."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "actor_state_dict": agent.actor.state_dict(),
        "meta": meta,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_actor_export(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=True)

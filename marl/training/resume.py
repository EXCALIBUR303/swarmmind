"""Trainer-state snapshot/restore for bitwise-exact resume (§33.3).

Checkpointing the network/optimizer/ValueNorm state alone is not enough
for exact resume: the NEXT rollout's data depends on (a) exactly where in
each seeded RNG stream training currently is, and (b) the exact live
state of every parallel environment (mid-episode robot positions,
beliefs, etc.) — checkpoints happen at update boundaries, not episode
boundaries, so an env can be mid-episode when a checkpoint is taken.
Both are captured here and folded into the checkpoint's `extra_rng_state`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from marl.envs.core.types import OBSTACLE
from marl.envs.vector import SyncVectorSwarm


def snapshot_trainer_state(
    env_seed_gens: list[np.random.Generator],
    policy_gen: torch.Generator,
    shuffle_rng: np.random.Generator,
    vec_env: SyncVectorSwarm,
) -> dict[str, Any]:
    return {
        "env_seed_gen_states": [g.bit_generator.state for g in env_seed_gens],
        "policy_gen_state": policy_gen.get_state(),
        "shuffle_rng_state": shuffle_rng.bit_generator.state,
        "vec_env_world_states": [w.state for w in vec_env.worlds],
    }


def restore_trainer_state(
    snapshot: dict[str, Any],
    env_seed_gens: list[np.random.Generator],
    policy_gen: torch.Generator,
    shuffle_rng: np.random.Generator,
    vec_env: SyncVectorSwarm,
) -> None:
    for gen, state in zip(env_seed_gens, snapshot["env_seed_gen_states"], strict=True):
        gen.bit_generator.state = state
    policy_gen.set_state(snapshot["policy_gen_state"])
    shuffle_rng.bit_generator.state = snapshot["shuffle_rng_state"]

    for world, state in zip(vec_env.worlds, snapshot["vec_env_world_states"], strict=True):
        world.state = state
        world._n_reachable_free = int(np.sum(state.terrain != OBSTACLE))

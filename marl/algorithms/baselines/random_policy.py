"""Random policy: the non-learning baseline (§14.1).

Each robot samples uniformly from its valid (masked) actions, including
WAIT. Under `action_masking: false` at the environment level, the mask is
all-ones, so this degenerates to sampling uniformly over all 5 actions.
"""

from __future__ import annotations

import numpy as np

from marl.envs.core.types import WAIT
from marl.envs.core.world import SwarmWorld


class RandomPolicy:
    """Implements the `Policy` protocol. Stateless; has no training."""

    def act(self, world: SwarmWorld, rng: np.random.Generator) -> np.ndarray:
        state = world.state
        if state is None:
            raise RuntimeError("RandomPolicy.act() called on a world before reset()")
        _, _, masks = world.all_actor_obs()
        actions = np.full(state.n_robots, WAIT, dtype=np.int32)
        for i in range(state.n_robots):
            valid = np.where(masks[i] == 1)[0]
            if len(valid) > 0:
                actions[i] = int(rng.choice(valid))
        return actions

"""The shared `Policy` interface every algorithm implements (Random,
heuristic reference, and — from Phase 4 onward — IPPO/MAPPO actors).

A `Policy` acts on a whole `SwarmWorld`, not just a raw observation array,
because the reference heuristic (§14.2) needs full-belief BFS pathfinding
that a windowed CNN observation doesn't carry. This does NOT weaken the
CTDE information boundary: every `Policy.act()` implementation must still
only read information a decentralized actor is entitled to — i.e.
`world.state.beliefs[i]` / `world.actor_obs(i)` per robot, never privileged
world-truth. This is enforced for the reference implementations here by
construction, and will be enforced for learned policies (Phase 4/5) by the
import-boundary tests and the §12.4 counterfactual checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from marl.envs.core.world import SwarmWorld


class Policy(Protocol):
    """Any policy capable of producing a joint action for one `SwarmWorld`."""

    def act(self, world: SwarmWorld, rng: np.random.Generator) -> np.ndarray:
        """Return int32[N] joint actions given the world's CURRENT state.

        `rng` is the caller's seeded generator — implementations must use
        it (never the numpy global RNG or their own unseeded source) so
        that evaluation is reproducible given an evaluation seed (§22, §34).
        """
        ...

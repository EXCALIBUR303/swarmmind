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

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

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


def load_actor(path: str, map_location: str = "cpu") -> tuple[Any, dict]:
    """Load an actor-only export (§12.3, §15.10) and return
    `(ActorPolicy, meta)`. Uses `weights_only=True` — this file, and this
    function, NEVER touch critic code, by construction: it is impossible
    to run a critic forward pass from what this function returns."""
    from marl.networks.actor import ActorPolicy
    from marl.training.checkpoint import load_actor_export

    payload = load_actor_export(path, map_location=map_location)
    meta = payload["meta"]
    grid_size = meta.get("grid_size", 11)
    hidden = meta.get("hidden", 256)
    actor = ActorPolicy(grid_size=grid_size, hidden=hidden)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return actor, meta


class LearnedActorPolicy:
    """Wraps an `ActorPolicy` (a trained IPPO/MAPPO actor — MAPPO's actor
    is architecturally identical to IPPO's, since the critic is the only
    difference between them, §16.3) so it can be driven through the same
    `Policy` protocol as Random/heuristic — used by the evaluator, replay
    recorder, and (from Phase 10) the backend's live sessions.

    Only ever calls `world.all_actor_obs()` — the actor's local,
    non-privileged view. Never touches `world.state` directly, never
    imports anything from `global_state` or `critics`.
    """

    def __init__(self, actor: Any, stochastic: bool = True) -> None:
        self.actor = actor
        self.stochastic = stochastic
        self._generator = torch.Generator()

    def _prepare_grid(self, grids: np.ndarray) -> np.ndarray:
        """Hook for evaluation-time input ablation (§22.4 item 4):
        `AblatedActorPolicy` overrides this to zero one observation channel
        before the actor ever sees it. Identity here."""
        return grids

    def act(self, world: SwarmWorld, rng: np.random.Generator) -> np.ndarray:
        # Re-seed the torch generator from the caller's numpy `rng` on
        # every call, so the whole episode's action sequence is
        # reproducible given the same `rng` stream (§22, §34) — torch and
        # numpy generators can't share state directly, so this is the
        # bridge between them.
        self._generator.manual_seed(int(rng.integers(0, 2**31 - 1)))
        grids, vecs, masks = world.all_actor_obs()
        grid_t = torch.as_tensor(self._prepare_grid(grids), dtype=torch.float32)
        vec_t = torch.as_tensor(vecs, dtype=torch.float32)
        mask_t = torch.as_tensor(masks, dtype=torch.bool)
        actions, _ = self.actor.act(
            grid_t, vec_t, mask_t, generator=self._generator, stochastic=self.stochastic
        )
        return actions.cpu().numpy().astype(np.int32)


class AblatedActorPolicy(LearnedActorPolicy):
    """`LearnedActorPolicy`, but with one local-observation channel zeroed
    before the actor sees it — EVALUATION ONLY, never during training
    (§22.4 item 4: "genuine learning" evidence). Zeroing channel 5
    (`visited_known`, the team's shared visited cells under comm mode B) or
    channel 6 (`teammate`, currently-sensed teammates) removes information
    a coordinating policy would use; if doing so changes outcomes
    measurably, the policy was actually using that channel rather than
    ignoring it."""

    def __init__(self, actor: Any, channel: int, stochastic: bool = True) -> None:
        super().__init__(actor, stochastic=stochastic)
        self.channel = channel

    def _prepare_grid(self, grids: np.ndarray) -> np.ndarray:
        grids = grids.copy()
        grids[..., self.channel, :, :] = 0
        return grids

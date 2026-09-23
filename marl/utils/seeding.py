"""Reproducibility utilities: global seeding and independent RNG streams.

See SWARMMIND_TECHNICAL_SPEC.md §34 (Reproducibility).

Rules enforced here:
  - `seed_everything` seeds every global RNG a third-party library might touch
    (python's `random`, numpy's legacy global state, torch CPU/CUDA).
  - Environment code MUST NOT use the numpy global RNG or python's `random`
    module directly. It must be constructed with its own `np.random.Generator`
    drawn from a `SeedStreams` instance. This is enforced by
    `tests/test_import_boundaries.py` (a grep-based check).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a required dependency
    torch = None  # type: ignore[assignment]


def seed_everything(seed: int) -> None:
    """Seed every global RNG this process might touch.

    This does NOT seed any `SeedStreams`-derived generator; those are
    independent and must be constructed explicitly via `make_streams`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # legacy global generator, for third-party code only
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass
class SeedStreams:
    """Independent, reproducible RNG streams spawned from one root seed.

    Each stream is derived via `np.random.SeedSequence.spawn`, which
    guarantees statistical independence between streams even though they
    share a common root. This lets a training run reproduce exactly while
    still giving every consumer (each vectorized env slot, the policy's
    action sampler, network init) its own uncorrelated stream.
    """

    root_seed: int
    _seq: np.random.SeedSequence = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._seq = np.random.SeedSequence(self.root_seed)

    def spawn_generators(self, n: int) -> list[np.random.Generator]:
        """Spawn `n` independent `np.random.Generator` instances."""
        children = self._seq.spawn(n)
        return [np.random.default_rng(c) for c in children]

    def spawn_env_seeds(self, n_envs: int) -> list[int]:
        """Spawn `n_envs` independent integer seeds for vectorized env slots.

        Each returned seed is itself the root of that env slot's own
        SeedSequence-derived map-generation stream (see
        `marl.envs.core.mapgen`), so training map draws across envs never
        collide or correlate.
        """
        children = self._seq.spawn(n_envs)
        # Derive a single high-entropy integer per child stream.
        return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

    def spawn_torch_generator(self) -> torch.Generator:
        """Spawn a torch.Generator (CPU) for policy action sampling."""
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is not installed")
        child = self._seq.spawn(1)[0]
        seed_int = int(child.generate_state(1, dtype=np.uint32)[0])
        gen = torch.Generator()
        gen.manual_seed(seed_int)
        return gen

    def spawn_init_seed(self) -> int:
        """Spawn a single integer seed for network weight initialization."""
        child = self._seq.spawn(1)[0]
        return int(child.generate_state(1, dtype=np.uint32)[0])


def make_streams(seed: int) -> SeedStreams:
    """Convenience constructor matching the spec's `make_streams(seed)` name."""
    return SeedStreams(root_seed=seed)

"""Orthogonal weight initialization helpers (§15.2).

Conv/hidden layers: gain sqrt(2). Policy head: gain 0.01 (near-uniform
initial policy over valid actions). Value head: gain 1.0. Biases: 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

GAIN_HIDDEN = 2.0**0.5
GAIN_POLICY_HEAD = 0.01
GAIN_VALUE_HEAD = 1.0


def orthogonal_init(module: nn.Module, gain: float) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


def init_hidden_(module: nn.Module) -> None:
    """Apply hidden-layer (Conv/Linear) orthogonal init with gain sqrt(2)
    to every Conv2d/Linear submodule of `module`."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            orthogonal_init(m, GAIN_HIDDEN)


def init_head_(module: nn.Module, gain: float) -> None:
    """Apply orthogonal init with the given gain to a single head layer
    (e.g. the final Linear of a policy or value head)."""
    orthogonal_init(module, gain)


def seeded_context(seed: int):
    """Context manager: run network construction under a fixed torch seed,
    so weight init is reproducible given `SeedStreams.spawn_init_seed()`."""

    class _Ctx:
        def __enter__(self_inner):
            self_inner.state = torch.get_rng_state()
            torch.manual_seed(seed)
            return self_inner

        def __exit__(self_inner, *exc):
            torch.set_rng_state(self_inner.state)
            return False

    return _Ctx()

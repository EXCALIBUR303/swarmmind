"""Masked categorical action distribution (§15.3).

Masked logits use a large finite negative value (-1e9), never -inf, to
avoid NaN propagating through entropy (`-inf * 0 = nan`) — see §16.8 item 9.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

MASK_FILL_VALUE = -1.0e9


class MaskedCategorical:
    """A `Categorical` distribution over logits with invalid actions masked out.

    `mask` is int8/bool [B, n_actions]; 1 = valid, 0 = invalid. Every
    forward pass MUST be given a mask (there is no "no mask" mode here —
    callers with `action_masking: false` at the env level simply pass an
    all-ones mask, which is what the environment itself already returns
    in that case, per §10.2).
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        masked_logits = logits.masked_fill(~mask, MASK_FILL_VALUE)
        self._dist = Categorical(logits=masked_logits)
        self.mask = mask

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            return self._dist.sample()
        # torch.distributions.Categorical.sample doesn't take a generator
        # directly; sample via the underlying probs + torch.multinomial,
        # which does, so action sampling is reproducible given `generator`.
        probs = self._dist.probs
        idx = torch.multinomial(probs, num_samples=1, generator=generator)
        return idx.squeeze(-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    @property
    def probs(self) -> torch.Tensor:
        return self._dist.probs

    def mode(self) -> torch.Tensor:
        """Greedy (argmax) action — used for deterministic/greedy evaluation."""
        return torch.argmax(self._dist.logits, dim=-1)

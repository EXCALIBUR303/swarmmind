"""Sensing geometry: disc offsets for robot sensors (§11.2).

v1 has no occlusion (`sensing.occlusion: false` is the only supported mode
in Phase 2; the line-of-sight stretch option is a documented future
extension, spec §11.2).

Offsets are precomputed once per (radius) and cached, since they are
identical for every robot and every step — only the center moves.
"""

from __future__ import annotations

from functools import cache

import numpy as np


@cache
def _disc_offsets(radius_squared_ceil: int, radius: float) -> tuple[tuple[int, int], ...]:
    """Integer (dy, dx) offsets with dy^2+dx^2 <= radius^2, radius-agnostic cache key."""
    r_int = int(np.ceil(radius))
    offsets = []
    r2 = radius * radius
    for dy in range(-r_int, r_int + 1):
        for dx in range(-r_int, r_int + 1):
            if dy * dy + dx * dx <= r2:
                offsets.append((dy, dx))
    return tuple(offsets)


def disc_offsets(radius: float) -> np.ndarray:
    """Return int32[K, 2] array of (dy, dx) offsets within `radius` (Euclidean)."""
    offs = _disc_offsets(int(np.ceil(radius * radius)) + 1, radius)
    return np.array(offs, dtype=np.int32)


def sensed_cells(
    center: tuple[int, int], radius: float, height: int, width: int
) -> np.ndarray:
    """In-bounds cells within Euclidean `radius` of `center` (§11.2).

    Returns an int32[K, 2] array of (y, x) coordinates. No occlusion (v1).
    """
    cy, cx = center
    offsets = disc_offsets(radius)
    ys = offsets[:, 0] + cy
    xs = offsets[:, 1] + cx
    mask = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    return np.stack([ys[mask], xs[mask]], axis=1)


def sensed_mask(center: tuple[int, int], radius: float, height: int, width: int) -> np.ndarray:
    """Boolean [H, W] mask of sensed cells (convenience wrapper over `sensed_cells`)."""
    mask = np.zeros((height, width), dtype=bool)
    cells = sensed_cells(center, radius, height, width)
    if len(cells) > 0:
        mask[cells[:, 0], cells[:, 1]] = True
    return mask

"""Procedural map generation (§20.1).

Deterministic given (level config, seed). Every generated map is guaranteed
reachable: obstacles that would isolate part of the grid are filled in
("pocket filling"), and every target is BFS-reachable from every robot
start by construction (checked, and retried on failure).

Uses ONLY a local `np.random.Generator` — never the numpy global RNG
(enforced by `tests/test_import_boundaries.py`).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.types import FREE, OBSTACLE

MAX_ATTEMPTS = 100
MIN_COMPONENT_FRACTION = 0.60


class MapGenerationError(RuntimeError):
    """Raised when no valid map could be generated within MAX_ATTEMPTS."""


def _hash64(seed: int, attempt: int) -> int:
    """Deterministic per-attempt seed derivation (§20.1 step 6)."""
    seq = np.random.SeedSequence([seed, attempt])
    return int(seq.generate_state(1, dtype=np.uint64)[0] % (2**31))


def _largest_component_mask(free: np.ndarray) -> np.ndarray:
    """4-connected largest component of `free` (bool[H,W]). BFS-based."""
    height, width = free.shape
    visited = np.zeros_like(free, dtype=bool)
    best_mask = np.zeros_like(free, dtype=bool)
    best_size = 0

    for sy in range(height):
        for sx in range(width):
            if not free[sy, sx] or visited[sy, sx]:
                continue
            # BFS this component
            comp_cells = []
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            while q:
                y, x = q.popleft()
                comp_cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and free[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(comp_cells) > best_size:
                best_size = len(comp_cells)
                best_mask = np.zeros_like(free, dtype=bool)
                for y, x in comp_cells:
                    best_mask[y, x] = True

    return best_mask


def _bfs_distances(free: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """4-connected BFS distances from `source` over `free`. -1 = unreachable."""
    height, width = free.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    sy, sx = source
    if not free[sy, sx]:
        return dist
    dist[sy, sx] = 0
    q = deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and free[ny, nx] and dist[ny, nx] == -1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))
    return dist


def _try_generate(cfg: EnvConfig, rng: np.random.Generator) -> MapSpec | None:
    height, width = cfg.height, cfg.width

    # 1-2: obstacles
    obstacle_mask = rng.random((height, width)) < cfg.obstacle_density
    terrain = np.where(obstacle_mask, OBSTACLE, FREE).astype(np.int8)

    # 3: connectivity via pocket filling
    free_mask = terrain == FREE
    component = _largest_component_mask(free_mask)
    if component.sum() < MIN_COMPONENT_FRACTION * (height * width):
        return None
    terrain = np.where(component, FREE, OBSTACLE).astype(np.int8)
    free_mask = component

    free_cells = list(zip(*np.where(free_mask), strict=True))
    free_cells = [(int(y), int(x)) for y, x in free_cells]
    if len(free_cells) < cfg.n_robots + cfg.n_targets:
        return None

    # 4: spawn
    if cfg.spawn == "clustered":
        anchor_idx = rng.integers(0, len(free_cells))
        anchor = free_cells[anchor_idx]
        dist_from_anchor = _bfs_distances(free_mask, anchor)
        candidates = [(dist_from_anchor[y, x], y, x) for y, x in free_cells if dist_from_anchor[y, x] >= 0]
        candidates.sort()  # nearest first; ties by (y, x) — lexicographic, deterministic
        if len(candidates) < cfg.n_robots:
            return None
        robot_starts = [(y, x) for _, y, x in candidates[: cfg.n_robots]]
    else:  # scattered
        idxs = rng.choice(len(free_cells), size=cfg.n_robots, replace=False)
        robot_starts = [free_cells[i] for i in idxs]

    # 5: targets, BFS distance >= target_min_dist from EVERY start
    start_dists = [_bfs_distances(free_mask, s) for s in robot_starts]
    target_candidates = []
    used = set(robot_starts)
    for y, x in free_cells:
        if (y, x) in used:
            continue
        if all(d[y, x] >= cfg.target_min_dist for d in start_dists):
            target_candidates.append((y, x))
    if len(target_candidates) < cfg.n_targets:
        return None
    perm = rng.permutation(len(target_candidates))
    targets = [target_candidates[i] for i in perm[: cfg.n_targets]]

    # 6: validation — every target reachable from every start
    for d in start_dists:
        for ty, tx in targets:
            if d[ty, tx] < 0:
                return None

    return MapSpec.from_terrain_array(
        id="pending",  # filled in by caller with seed/attempt info
        terrain=terrain,
        robot_starts=robot_starts,
        targets=targets,
        generator={"kind": "procedural", "level": cfg.name},
    )


def generate_map(cfg: EnvConfig, seed: int, id_prefix: str | None = None) -> MapSpec:
    """Generate one valid MapSpec for `cfg`, deterministic given `seed` (§20.1).

    Raises `MapGenerationError` after `MAX_ATTEMPTS` failed attempts —
    this means the level config itself is infeasible (too dense, too many
    robots/targets for the grid size), not a transient failure.
    """
    for attempt in range(MAX_ATTEMPTS):
        attempt_seed = _hash64(seed, attempt)
        rng = np.random.default_rng(attempt_seed)
        spec = _try_generate(cfg, rng)
        if spec is not None:
            map_id = f"{id_prefix or cfg.name}-s{seed}-a{attempt}"
            spec.id = map_id
            spec.generator.update({"level": cfg.name, "seed": seed, "attempt": attempt})
            return spec
    raise MapGenerationError(
        f"could not generate a valid map for level '{cfg.name}' (seed={seed}) "
        f"after {MAX_ATTEMPTS} attempts; the level config may be infeasible "
        f"(width={cfg.width}, height={cfg.height}, n_robots={cfg.n_robots}, "
        f"n_targets={cfg.n_targets}, obstacle_density={cfg.obstacle_density}, "
        f"target_min_dist={cfg.target_min_dist})"
    )

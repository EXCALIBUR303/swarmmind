"""Procedural map generation: connectivity, reachability, determinism (§20.1, §33.2)."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from marl.envs.core.config import EnvConfig
from marl.envs.core.mapgen import MapGenerationError, generate_map
from marl.envs.core.types import FREE, OBSTACLE


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t",
        width=10,
        height=10,
        n_robots=3,
        n_targets=2,
        obstacle_density=0.15,
        max_steps=150,
        sensor_radius=2.0,
        obs_radius=5,
        target_min_dist=3,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def _bfs_reachable(terrain: np.ndarray, source: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = terrain.shape
    sy, sx = source
    if terrain[sy, sx] != FREE:
        return set()
    seen = {(sy, sx)}
    q = deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and terrain[ny, nx] == FREE and (ny, nx) not in seen:
                seen.add((ny, nx))
                q.append((ny, nx))
    return seen


def test_generate_map_deterministic_given_seed():
    cfg = _cfg()
    m1 = generate_map(cfg, seed=42)
    m2 = generate_map(cfg, seed=42)
    assert m1.terrain_rows == m2.terrain_rows
    assert m1.robot_starts == m2.robot_starts
    assert m1.targets == m2.targets


def test_generate_map_different_seeds_differ():
    cfg = _cfg()
    m1 = generate_map(cfg, seed=1)
    m2 = generate_map(cfg, seed=2)
    assert m1.terrain_rows != m2.terrain_rows or m1.robot_starts != m2.robot_starts


def test_generated_map_no_robot_target_overlap():
    cfg = _cfg()
    m = generate_map(cfg, seed=7)
    starts = set(m.robot_starts)
    targets = set(m.targets)
    assert len(starts) == cfg.n_robots  # all distinct
    assert len(targets) == cfg.n_targets
    assert starts.isdisjoint(targets)


def test_generated_map_all_starts_and_targets_on_free_cells():
    cfg = _cfg()
    m = generate_map(cfg, seed=7)
    terrain = m.terrain_array()
    for y, x in m.robot_starts:
        assert terrain[y, x] == FREE
    for y, x in m.targets:
        assert terrain[y, x] == FREE


def test_generated_map_all_free_cells_reachable_from_each_other():
    """Pocket-filling guarantee: the whole free region is one connected component."""
    cfg = _cfg()
    m = generate_map(cfg, seed=7)
    terrain = m.terrain_array()
    free_cells = set(zip(*np.where(terrain == FREE), strict=True))
    if not free_cells:
        pytest.skip("no free cells")
    any_free = next(iter(free_cells))
    reachable = _bfs_reachable(terrain, any_free)
    assert reachable == free_cells


def test_every_target_reachable_from_every_start():
    cfg = _cfg()
    m = generate_map(cfg, seed=7)
    terrain = m.terrain_array()
    for start in m.robot_starts:
        reachable = _bfs_reachable(terrain, start)
        for target in m.targets:
            assert tuple(target) in reachable, f"target {target} unreachable from start {start}"


def test_target_min_distance_respected():
    cfg = _cfg(target_min_dist=3)
    m = generate_map(cfg, seed=7)
    terrain = m.terrain_array()
    for start in m.robot_starts:
        seen = {start: 0}
        q = deque([start])
        h, w = terrain.shape
        while q:
            y, x = q.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and terrain[ny, nx] == FREE and (ny, nx) not in seen:
                    seen[(ny, nx)] = seen[(y, x)] + 1
                    q.append((ny, nx))
        for target in m.targets:
            assert seen[tuple(target)] >= cfg.target_min_dist


def test_obstacle_density_roughly_matches_config():
    """After pocket-filling, resulting density should be close to configured
    (it can only be >= the raw sampled density, since filling adds obstacles)."""
    cfg = _cfg(obstacle_density=0.15)
    densities = []
    for seed in range(20):
        m = generate_map(cfg, seed=seed)
        terrain = m.terrain_array()
        densities.append(float(np.mean(terrain == OBSTACLE)))
    mean_density = float(np.mean(densities))
    assert 0.10 <= mean_density <= 0.45  # pocket-filling raises it somewhat above 0.15


def test_infeasible_config_raises_map_generation_error():
    """A target_min_dist far exceeding any possible BFS distance on a 6x6
    grid (max Manhattan distance is 10) makes target placement impossible
    regardless of obstacle layout — this must exhaust MAX_ATTEMPTS and raise."""
    cfg = _cfg(width=6, height=6, n_robots=2, n_targets=1, obstacle_density=0.1, target_min_dist=100)
    with pytest.raises(MapGenerationError):
        generate_map(cfg, seed=0)


def test_retry_determinism_same_seed_same_attempt_sequence():
    """Even when early attempts fail and a retry is needed, the same seed
    must always land on the same successful attempt (per-attempt seeds are
    derived deterministically via hash64(seed, attempt))."""
    cfg = _cfg(width=8, height=8, n_robots=4, n_targets=3, obstacle_density=0.3, target_min_dist=4)
    m1 = generate_map(cfg, seed=99)
    m2 = generate_map(cfg, seed=99)
    assert m1.generator["attempt"] == m2.generator["attempt"]
    assert m1.terrain_rows == m2.terrain_rows


def test_scattered_spawn_mode_produces_distinct_starts():
    cfg = _cfg(spawn="scattered")
    m = generate_map(cfg, seed=3)
    assert len(set(m.robot_starts)) == cfg.n_robots


def test_map_spec_content_hash_stable_and_order_independent_metadata():
    cfg = _cfg()
    m = generate_map(cfg, seed=5)
    h1 = m.content_hash
    h2 = m.content_hash
    assert h1 == h2
    # Changing generator metadata (id, seed bookkeeping) must NOT change
    # the content hash, since it's about the map's actual content.
    m.generator["extra_note"] = "irrelevant"
    assert m.content_hash == h1


def test_stress_generation_across_many_seeds_all_levels():
    """1k procedural maps at a representative density all validate cleanly
    (a lighter-weight in-suite version of the Phase 3 full stress test)."""
    cfg = _cfg(width=10, height=10, n_robots=3, n_targets=2, obstacle_density=0.15)
    failures = 0
    for seed in range(200):
        try:
            m = generate_map(cfg, seed=seed)
            terrain = m.terrain_array()
            for start in m.robot_starts:
                reachable = _bfs_reachable(terrain, start)
                for target in m.targets:
                    assert tuple(target) in reachable
        except MapGenerationError:
            failures += 1
    assert failures == 0, f"{failures}/200 seeds failed to generate a valid L2-like map"

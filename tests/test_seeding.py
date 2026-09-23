"""Seeding reproducibility tests (§34)."""

from __future__ import annotations

import random

import numpy as np

from marl.utils.seeding import SeedStreams, make_streams, seed_everything


def test_seed_everything_reproducible_python_random():
    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_reproducible_numpy_legacy():
    seed_everything(42)
    a = np.random.rand(5).tolist()
    seed_everything(42)
    b = np.random.rand(5).tolist()
    assert a == b


def test_seed_streams_reproducible():
    s1 = make_streams(7)
    s2 = make_streams(7)
    gens1 = s1.spawn_generators(3)
    gens2 = s2.spawn_generators(3)
    for g1, g2 in zip(gens1, gens2, strict=True):
        assert g1.random(5).tolist() == g2.random(5).tolist()


def test_seed_streams_independent_across_children():
    streams = make_streams(7)
    gens = streams.spawn_generators(2)
    a = gens[0].random(10)
    b = gens[1].random(10)
    assert not np.allclose(a, b)


def test_seed_streams_different_root_seeds_differ():
    a = make_streams(1).spawn_generators(1)[0].random(10)
    b = make_streams(2).spawn_generators(1)[0].random(10)
    assert not np.allclose(a, b)


def test_spawn_env_seeds_deterministic_and_unique():
    s1 = make_streams(99)
    s2 = make_streams(99)
    seeds1 = s1.spawn_env_seeds(8)
    seeds2 = s2.spawn_env_seeds(8)
    assert seeds1 == seeds2
    assert len(set(seeds1)) == len(seeds1)


def test_spawn_init_seed_deterministic():
    s1 = make_streams(5)
    s2 = make_streams(5)
    assert s1.spawn_init_seed() == s2.spawn_init_seed()


def test_seed_streams_is_seed_streams_instance():
    assert isinstance(make_streams(1), SeedStreams)

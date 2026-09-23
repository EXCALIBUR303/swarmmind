"""Tests for path safety (§37) and canonical hashing (§32, §20.2)."""

from __future__ import annotations

import pytest

from marl.utils.hashing import canonical_json, config_hash, object_hash, sha256_hex
from marl.utils.paths import UnsafePathError, is_safe_id, safe_join


def test_is_safe_id_accepts_normal_ids():
    assert is_safe_id("run_abc-123.v2")
    assert is_safe_id("L2")


def test_is_safe_id_rejects_traversal():
    assert not is_safe_id("../etc/passwd")
    assert not is_safe_id("a/b")
    assert not is_safe_id("")
    assert not is_safe_id("a" * 200)


def test_safe_join_rejects_unsafe_segment(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "../escape")


def test_safe_join_stays_within_root(tmp_path):
    result = safe_join(tmp_path, "run_1", "checkpoints")
    assert result.is_relative_to(tmp_path.resolve())


def test_canonical_json_key_order_independent():
    a = canonical_json({"b": 1, "a": 2})
    b = canonical_json({"a": 2, "b": 1})
    assert a == b


def test_config_hash_stable_length_and_determinism():
    cfg = {"algo": "mappo", "seed": 3}
    h1 = config_hash(cfg)
    h2 = config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 8


def test_config_hash_changes_with_content():
    h1 = config_hash({"seed": 1})
    h2 = config_hash({"seed": 2})
    assert h1 != h2


def test_object_hash_full_length():
    assert len(object_hash({"x": 1})) == 64


def test_sha256_hex_known_value():
    assert sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

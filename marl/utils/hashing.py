"""Canonical hashing utilities.

Used for: config_hash (run naming, §32), map hashes (MapSpec, §20.2), and the
eval-set immutability manifest (§20.3).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize `obj` to a canonical JSON string.

    Keys are sorted and separators are minimal, so semantically identical
    objects always hash identically regardless of construction order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(resolved_config: dict[str, Any], length: int = 8) -> str:
    """Short, stable hash of a resolved config dict, for run directory names."""
    return sha256_hex(canonical_json(resolved_config))[:length]


def object_hash(obj: Any) -> str:
    """Full sha256 hex digest of the canonical JSON form of `obj`."""
    return sha256_hex(canonical_json(obj))

"""Frozen validation/test map sets and the training-isolation guard (§20.3).

Validation and test maps are generated ONCE (by `scripts/make_eval_maps.py`)
and saved as JSON with their content hashes. They are then loaded from
disk everywhere else — never regenerated on the fly — so that "the test
set" means the same 100 maps in every run, forever.

Seed ranges (§20.3): validation = 1,000,000 + i; test = 2,000,000 + i.
These are large offsets from the unbounded training-seed range specifically
so that an accidental small training seed can never collide with them.
"""

from __future__ import annotations

import json
from pathlib import Path

from marl.envs.core.config import EnvConfig
from marl.envs.core.mapgen import generate_map
from marl.envs.core.maps_io import MapSpec
from marl.utils.paths import MAPS_DIR

VALIDATION_SEED_BASE = 1_000_000
TEST_SEED_BASE = 2_000_000


def generate_validation_maps(cfg: EnvConfig, n: int = 50) -> list[MapSpec]:
    return [generate_map(cfg, seed=VALIDATION_SEED_BASE + i, id_prefix=f"{cfg.name}-val") for i in range(n)]


def generate_test_maps(cfg: EnvConfig, n: int = 100) -> list[MapSpec]:
    return [generate_map(cfg, seed=TEST_SEED_BASE + i, id_prefix=f"{cfg.name}-test") for i in range(n)]


def _split_dir(split: str, base_dir: Path | None = None) -> Path:
    root = base_dir or MAPS_DIR
    return root / split


def save_eval_set(maps: list[MapSpec], level: str, split: str, base_dir: Path | None = None) -> Path:
    """Save `maps` as `<base_dir>/<split>/<level>.json`, with content hashes."""
    path = _split_dir(split, base_dir) / f"{level}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "level": level,
        "split": split,
        "n_maps": len(maps),
        "maps": [m.to_dict() for m in maps],
        "hashes": [m.content_hash for m in maps],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_eval_set(level: str, split: str, base_dir: Path | None = None) -> list[MapSpec]:
    path = _split_dir(split, base_dir) / f"{level}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen {split} set for level '{level}' not found at {path}; "
            "run scripts/make_eval_maps.py first"
        )
    with path.open("r") as f:
        payload = json.load(f)
    return [
        MapSpec(
            id=m["id"],
            width=m["width"],
            height=m["height"],
            terrain_rows=m["terrain"],
            robot_starts=[tuple(p) for p in m["robot_starts"]],
            targets=[tuple(p) for p in m["targets"]],
            generator=m.get("generator", {}),
        )
        for m in payload["maps"]
    ]


def eval_set_exists(level: str, split: str, base_dir: Path | None = None) -> bool:
    return _split_dir(split, base_dir).joinpath(f"{level}.json").is_file()


def frozen_hash_manifest(levels: list[str], base_dir: Path | None = None) -> set[str]:
    """All content hashes across validation+test sets, for every level in
    `levels` that has a frozen set on disk. Used by the training-map
    generator to reject any training map matching a frozen eval map
    (§20.3 isolation guard)."""
    hashes: set[str] = set()
    for level in levels:
        for split in ("validation", "test"):
            if eval_set_exists(level, split, base_dir):
                for m in load_eval_set(level, split, base_dir):
                    hashes.add(m.content_hash)
    return hashes


def is_eval_map(map_spec: MapSpec, manifest: set[str]) -> bool:
    return map_spec.content_hash in manifest

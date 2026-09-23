"""Replay load/save and the re-simulation determinism check (§27.2).

`resimulate_matches` re-plays a recorded replay's exact (map, recorded
actions) through a fresh `SwarmWorld` and checks every frame matches —
proving replays are a faithful record, not an approximation.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.world import SwarmWorld
from marl.replay.schema import Replay


def save_replay(replay: Replay, path: str | Path) -> None:
    """Save as gzip-compressed JSON (§27.3: `.json.gz`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = replay.model_dump_json().encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(data)


def load_replay(path: str | Path) -> Replay:
    path = Path(path)
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
    else:
        with path.open("r") as f:
            data = json.load(f)
    return Replay.model_validate(data)


def resimulate_matches(replay: Replay, cfg: EnvConfig) -> bool:
    """Re-run `replay`'s recorded actions on its own map through a fresh
    `SwarmWorld` and check every frame's positions/rewards/flags match
    exactly (§27.2 determinism test)."""
    map_spec = MapSpec(
        id=replay.map.id,
        width=replay.map.width,
        height=replay.map.height,
        terrain_rows=replay.map.terrain,
        robot_starts=[(p[0], p[1]) for p in replay.map.robot_starts],
        targets=[(p[0], p[1]) for p in replay.map.targets],
    )
    world = SwarmWorld(cfg)
    state = world.reset(map_spec=map_spec)

    for frame in replay.frames:
        actions = np.array(frame.action, dtype=np.int32)
        result = world.step(actions)

        if state.robot_pos.tolist() != frame.pos:
            return False
        if result.moved.tolist() != frame.moved:
            return False
        if result.blocked_by_robot.tolist() != frame.blocked_by_robot:
            return False
        if result.bumped_wall.tolist() != frame.bumped_wall:
            return False
        if not np.allclose(result.reward, frame.reward, atol=1e-6):
            return False
        if sorted(result.discovered) != sorted(frame.discovered):
            return False
        if sorted([list(r) for r in result.rescued]) != sorted(frame.rescued):
            return False

    return True

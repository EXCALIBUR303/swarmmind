"""MapSpec: the serializable description of one map instance (§20.2).

A MapSpec is either procedurally generated (carries its generator seed) or
loaded from a frozen/handmade file. Its hash is used for the eval-set
isolation guard (§20.3): training rejects any map whose hash appears in
the frozen validation/test manifests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from marl.envs.core.types import FREE, OBSTACLE
from marl.utils.hashing import object_hash

FREE_CHAR = "."
OBSTACLE_CHAR = "#"


@dataclass
class MapSpec:
    """A concrete map instance: terrain + robot starts + targets (§20.2)."""

    id: str
    width: int
    height: int
    terrain_rows: list[str]  # each row a string of '.'/'#', len == width, count == height
    robot_starts: list[tuple[int, int]]  # (y, x)
    targets: list[tuple[int, int]]  # (y, x)
    generator: dict[str, Any] = field(default_factory=lambda: {"kind": "handmade"})

    def __post_init__(self) -> None:
        if len(self.terrain_rows) != self.height:
            raise ValueError(
                f"MapSpec '{self.id}': {len(self.terrain_rows)} terrain rows != height {self.height}"
            )
        for row in self.terrain_rows:
            if len(row) != self.width:
                raise ValueError(
                    f"MapSpec '{self.id}': row length {len(row)} != width {self.width}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "terrain": self.terrain_rows,
            "robot_starts": [list(p) for p in self.robot_starts],
            "targets": [list(p) for p in self.targets],
            "generator": self.generator,
        }

    @property
    def content_hash(self) -> str:
        """Hash over terrain + starts + targets only (NOT id/generator metadata),
        per §20.2, so two maps with identical content always collide in the
        eval-isolation check regardless of how they were produced."""
        payload = {
            "width": self.width,
            "height": self.height,
            "terrain": self.terrain_rows,
            "robot_starts": sorted(self.robot_starts),
            "targets": sorted(self.targets),
        }
        return object_hash(payload)

    def terrain_array(self) -> np.ndarray:
        arr = np.zeros((self.height, self.width), dtype=np.int8)
        for y, row in enumerate(self.terrain_rows):
            for x, ch in enumerate(row):
                arr[y, x] = OBSTACLE if ch == OBSTACLE_CHAR else FREE
        return arr

    @classmethod
    def from_terrain_array(
        cls,
        id: str,
        terrain: np.ndarray,
        robot_starts: list[tuple[int, int]],
        targets: list[tuple[int, int]],
        generator: dict[str, Any] | None = None,
    ) -> MapSpec:
        height, width = terrain.shape
        rows = []
        for y in range(height):
            rows.append("".join(OBSTACLE_CHAR if terrain[y, x] == OBSTACLE else FREE_CHAR for x in range(width)))
        return cls(
            id=id,
            width=width,
            height=height,
            terrain_rows=rows,
            robot_starts=robot_starts,
            targets=targets,
            generator=generator or {"kind": "handmade"},
        )


def save_map(spec: MapSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(spec.to_dict(), f, indent=2)


def load_map(path: str | Path) -> MapSpec:
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    return MapSpec(
        id=data["id"],
        width=data["width"],
        height=data["height"],
        terrain_rows=data["terrain"],
        robot_starts=[tuple(p) for p in data["robot_starts"]],
        targets=[tuple(p) for p in data["targets"]],
        generator=data.get("generator", {"kind": "handmade"}),
    )


def load_structured_map_txt(path: str | Path, map_id: str | None = None) -> MapSpec:
    """Load a handmade structured map from the `#`/`.`/`R`/`T` text format (§20.4)."""
    path = Path(path)
    lines = [line.rstrip("\n") for line in path.read_text().splitlines() if line.strip() != ""]
    height = len(lines)
    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise ValueError(f"structured map '{path}': inconsistent row widths")

    terrain_rows: list[str] = []
    robot_starts: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    for y, line in enumerate(lines):
        row_chars = []
        for x, ch in enumerate(line):
            if ch == "R":
                robot_starts.append((y, x))
                row_chars.append(FREE_CHAR)
            elif ch == "T":
                targets.append((y, x))
                row_chars.append(FREE_CHAR)
            elif ch == "#":
                row_chars.append(OBSTACLE_CHAR)
            else:
                row_chars.append(FREE_CHAR)
        terrain_rows.append("".join(row_chars))

    return MapSpec(
        id=map_id or path.stem,
        width=width,
        height=height,
        terrain_rows=terrain_rows,
        robot_starts=robot_starts,
        targets=targets,
        generator={"kind": "structured", "source": str(path)},
    )

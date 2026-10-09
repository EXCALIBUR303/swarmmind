"""Lists and serves replays (§27.3, §28.1). Path-confined: a replay id is
resolved only through the index built from `index.json` files, then
joined with `safe_join`."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.services.errors import NotFoundError
from backend.app.services.jsonio import read_json
from marl.utils.paths import UnsafePathError, is_safe_id, safe_join


class ReplayStore:
    def __init__(self, replays_dir: Path) -> None:
        self.replays_dir = replays_dir
        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] = {}

    def refresh(self) -> None:
        found: dict[str, dict[str, Any]] = {}
        if self.replays_dir.is_dir():
            for idx in sorted(self.replays_dir.glob("*/*/index.json")):
                run_dir, study = idx.parent, idx.parent.parent.name
                entries = read_json(idx)
                if not isinstance(entries, list) or not is_safe_id(run_dir.name):
                    continue
                for e in entries:
                    rid = e.get("replay_id") if isinstance(e, dict) else None
                    if not isinstance(rid, str) or not is_safe_id(rid) or rid in found:
                        continue
                    if (run_dir / f"{rid}.json.gz").is_file():
                        found[rid] = {**e, "study": study, "run_id": run_dir.name}
        with self._lock:
            self._index = found

    def list(
        self,
        run_id: str | None = None,
        map_id: str | None = None,
        study: str | None = None,
        algo: str | None = None,
    ) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            items = list(self._index.values())
        for key, want in (("run_id", run_id), ("map_id", map_id), ("study", study), ("algo", algo)):
            if want is not None:
                items = [i for i in items if i.get(key) == want]
        return items

    def path(self, replay_id: str) -> Path:
        with self._lock:
            entry = self._index.get(replay_id)
        if entry is None:
            self.refresh()
            with self._lock:
                entry = self._index.get(replay_id)
        if entry is None:
            raise NotFoundError(f"replay '{replay_id}' not found")
        try:
            path = safe_join(self.replays_dir, entry["study"], entry["run_id"], f"{replay_id}.json.gz")
        except UnsafePathError as exc:  # pragma: no cover - ids are validated at index time
            raise NotFoundError(f"replay '{replay_id}' not found") from exc
        if not path.is_file():
            raise NotFoundError(f"replay '{replay_id}' not found")
        return path

    def initial(self, replay_id: str) -> dict[str, Any]:
        """`{known_team}` right after reset. Returned as recorded when the replay
        has it; otherwise derived by re-running `SwarmWorld.reset` on the replay's
        own map and env config — the same deterministic code path that produced
        the replay — so old study replays need no re-recording."""
        from marl.envs.core.config import EnvConfig
        from marl.envs.core.maps_io import MapSpec
        from marl.envs.core.world import SwarmWorld
        from marl.replay.loader import load_replay

        replay = load_replay(self.path(replay_id))
        if replay.initial is not None:
            return {"known_team": replay.initial.known_team, "source": "recorded"}
        m = replay.map
        spec = MapSpec(
            id=m.id, width=m.width, height=m.height, terrain_rows=m.terrain,
            robot_starts=[(p[0], p[1]) for p in m.robot_starts], targets=[(p[0], p[1]) for p in m.targets],
        )
        state = SwarmWorld(EnvConfig.model_validate(replay.env.config)).reset(map_spec=spec)
        return {"known_team": np.argwhere(state.known_team).tolist(), "source": "derived"}

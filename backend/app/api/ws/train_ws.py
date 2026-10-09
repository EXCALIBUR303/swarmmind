"""WS /ws/training/{run_id} (§28.3): tails `metrics.jsonl` / `eval_val.jsonl`
and pushes new complete lines plus status changes. Polling fallback:
`GET /api/experiments/{run_id}/metrics?from_update=`."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.api.deps import ID_RE
from backend.app.api.ws.common import CLOSE_FORBIDDEN, CLOSE_NOT_FOUND, origin_allowed
from backend.app.services.errors import ServiceError
from backend.app.services.jsonio import read_json

router = APIRouter()
_ID = re.compile(ID_RE)
POLL_SECONDS = 0.5
TERMINAL = {"completed", "failed", "interrupted"}


class JsonlTail:
    """Incrementally reads complete lines; a half-written last line waits."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def read_new(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return []
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        out = []
        for line in chunk[: end + 1].splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out


@router.websocket("/ws/training/{run_id}")
async def train_ws(websocket: WebSocket, run_id: str, from_update: int = 0) -> None:
    svc = websocket.app.state.services
    if not origin_allowed(websocket, svc.settings.cors_origin):
        await websocket.close(code=CLOSE_FORBIDDEN)
        return
    if not _ID.match(run_id):
        await websocket.close(code=CLOSE_NOT_FOUND)
        return
    try:
        run_dir = svc.runs.run_dir(run_id)
    except ServiceError:
        await websocket.close(code=CLOSE_NOT_FOUND)
        return

    await websocket.accept()
    metrics, vals = JsonlTail(run_dir / "metrics.jsonl"), JsonlTail(run_dir / "eval_val.jsonl")
    last_status: str | None = None
    recv = asyncio.create_task(websocket.receive())
    try:
        while True:
            sent = 0
            for rec in metrics.read_new():
                if int(rec.get("update", 0)) >= from_update:
                    await websocket.send_json({"type": "update", **rec})
                    sent += 1
            for rec in vals.read_new():
                await websocket.send_json({"type": "val_eval", **rec})
                sent += 1
            manifest = read_json(run_dir / "manifest.json")
            status = manifest.get("status") if isinstance(manifest, dict) else "invalid"
            if status != last_status:
                await websocket.send_json({"type": "status", "status": status})
                last_status = status
            if status in TERMINAL and sent == 0:
                await websocket.send_json({"type": "end", "status": status})
                await websocket.close(code=1000)
                return
            done, _ = await asyncio.wait({recv}, timeout=POLL_SECONDS)
            if done and recv.result().get("type") == "websocket.disconnect":
                return
            if done:
                recv = asyncio.create_task(websocket.receive())  # ignore client chatter
    except WebSocketDisconnect:
        pass
    finally:
        recv.cancel()
        with contextlib.suppress(BaseException):
            await recv

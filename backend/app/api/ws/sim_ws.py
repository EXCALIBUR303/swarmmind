"""WS /ws/sessions/{id} (§28.3). The server owns the simulation clock; the
client only renders and sends commands."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.app.api.deps import ID_RE
from backend.app.api.ws.common import (
    CLOSE_FORBIDDEN,
    CLOSE_NOT_FOUND,
    CLOSE_REPLACED,
    origin_allowed,
    send_model,
)
from backend.app.schemas.sessions import (
    SessionReset,
    WsCommand,
    WsEpisodeEnd,
    WsError,
    WsFrameMessage,
    WsStateMessage,
)
from backend.app.services.errors import ServiceError
from backend.app.services.session_manager import Session

router = APIRouter()
logger = logging.getLogger(__name__)
_ID = re.compile(ID_RE)

MIN_SPEED, MAX_SPEED = 0.25, 8.0


class _Player:
    """Per-connection play state and clock."""

    def __init__(self, ws: WebSocket, session: Session, max_sps: int, base_sps: float) -> None:
        self.ws, self.session = ws, session
        self.max_sps, self.base_sps = max_sps, base_sps
        self.playing = False
        self.speed = 1.0
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def rate(self) -> float:
        return max(0.1, min(float(self.max_sps), self.base_sps * self.speed))

    async def send_state(self) -> None:
        with self.session.lock:
            snap = self.session.snapshot()
        await send_model(self.ws, WsStateMessage(playing=self.playing, speed=self.speed, snapshot=snap))

    async def send_error(self, code: str, message: str) -> None:
        await send_model(self.ws, WsError(code=code, message=message))

    async def emit(self, resp) -> None:
        for frame in resp.frames:
            await send_model(
                self.ws,
                WsFrameMessage(frame=frame, step_index=frame.t, metrics=resp.metrics),
            )
        if resp.done:
            self.playing = False
            fm = resp.final_metrics or {}
            await send_model(
                self.ws,
                WsEpisodeEnd(metrics=fm, success=bool(resp.metrics.success), steps=resp.step_index),
            )

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self.playing:
            t0 = loop.time()
            try:
                resp = await asyncio.to_thread(self.session.step, 1)
            except ServiceError as exc:
                self.playing = False
                await self.send_error("episode_done", exc.message)
                return
            except Exception:  # a policy/env fault must not leak details or kill the socket
                logger.exception("session %s: step failed", self.session.session_id)
                self.playing = False
                await self.send_error("internal_error", "the simulation step failed")
                return
            await self.emit(resp)
            if resp.done:
                return
            wait = max(0.0, 1.0 / self.rate() - (loop.time() - t0))
            self._wake.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=wait)

    def play(self) -> None:
        if self._task is None or self._task.done():
            self.playing = True
            self._task = asyncio.create_task(self._loop())

    async def pause(self) -> None:
        """Stop after the in-flight step has been sent (never cancel mid-step,
        or the server state would advance past what the client saw)."""
        self.playing = False
        self._wake.set()
        if self._task is not None:
            await self._task
            self._task = None


@router.websocket("/ws/sessions/{session_id}")
async def sim_ws(websocket: WebSocket, session_id: str) -> None:
    svc = websocket.app.state.services
    cfg = svc.settings
    if not origin_allowed(websocket, cfg.cors_origin):
        await websocket.close(code=CLOSE_FORBIDDEN)
        return
    if not _ID.match(session_id):
        await websocket.close(code=CLOSE_NOT_FOUND)
        return
    try:
        session = svc.sessions.get(session_id)
    except ServiceError:
        await websocket.close(code=CLOSE_NOT_FOUND)
        return

    await websocket.accept()
    # One driver per session: a new connection (e.g. a reconnect after a
    # half-open TCP drop) takes over; the previous one is closed with 4409.
    previous = session.owner
    session.owner = websocket
    if previous is not None:
        with contextlib.suppress(Exception):
            await previous.close(code=CLOSE_REPLACED)
    player = _Player(websocket, session, cfg.ws_max_steps_per_second, cfg.ws_base_steps_per_second)
    try:
        await player.send_state()
        while True:
            text = await websocket.receive_text()
            session.touch()
            if len(text.encode()) > cfg.ws_max_message_bytes:
                await player.send_error("message_too_large", "command exceeds the size limit")
                continue
            try:
                cmd = WsCommand.model_validate_json(text)
            except ValidationError:
                await player.send_error("bad_command", "expected {cmd: play|pause|step|reset|speed, value?}")
                continue
            await _handle(player, cmd, cfg)
    except WebSocketDisconnect:
        pass
    except (TimeoutError, RuntimeError):
        # Stalled or already-closed client: drop the connection.
        with contextlib.suppress(Exception):
            await websocket.close()
    finally:
        if session.owner is websocket:
            session.owner = None
        player.playing = False
        player._wake.set()
        if player._task is not None:
            with contextlib.suppress(Exception):
                await player._task


async def _handle(p: _Player, cmd: WsCommand, cfg: Any) -> None:
    if cmd.cmd == "play":
        if p.session.runner.done:
            await p.send_error("episode_done", "episode is finished; send reset")
            return
        p.play()
        await p.send_state()
    elif cmd.cmd == "pause":
        await p.pause()
        await p.send_state()
    elif cmd.cmd == "speed":
        if not isinstance(cmd.value, (int, float)) or isinstance(cmd.value, bool) or not MIN_SPEED <= cmd.value <= MAX_SPEED:
            await p.send_error("bad_value", f"speed must be a number in [{MIN_SPEED}, {MAX_SPEED}]")
            return
        p.speed = float(cmd.value)
        p._wake.set()
        await p.send_state()
    elif cmd.cmd == "step":
        if p.playing:
            await p.send_error("playing", "pause before stepping")
            return
        n = 1 if cmd.value is None else cmd.value
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= cfg.max_steps_per_request:
            await p.send_error("bad_value", f"step count must be an integer in [1, {cfg.max_steps_per_request}]")
            return
        try:
            resp = await asyncio.to_thread(p.session.step, n)
        except ServiceError as exc:
            await p.send_error("episode_done", exc.message)
            return
        await p.emit(resp)
    elif cmd.cmd == "reset":
        await p.pause()
        body = cmd.value if isinstance(cmd.value, dict) else {}
        try:
            req = SessionReset.model_validate(body)
            session = await asyncio.to_thread(_reset, p, req)
        except (ValidationError, ServiceError) as exc:
            await p.send_error("bad_reset", exc.message if isinstance(exc, ServiceError) else "invalid reset options")
            return
        p.session = session
        await p.send_state()


def _reset(p: _Player, req: SessionReset) -> Session:
    svc = p.ws.app.state.services
    return svc.sessions.reset(p.session.session_id, req)

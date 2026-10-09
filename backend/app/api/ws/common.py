"""WebSocket helpers: origin check (CORS does not cover WS) and bounded sends."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket
from pydantic import BaseModel

SEND_TIMEOUT_SECONDS = 10.0

# Application close codes (4000-4999 are free for apps).
CLOSE_FORBIDDEN = 4403
CLOSE_NOT_FOUND = 4404
CLOSE_REPLACED = 4409  # another connection took over this session


def origin_allowed(ws: WebSocket, allowed_origin: str) -> bool:
    """Browsers always send Origin; a missing Origin means a non-browser client
    (tests, curl), which the localhost bind already scopes."""
    origin = ws.headers.get("origin")
    return origin is None or origin == allowed_origin


async def send_model(ws: WebSocket, msg: BaseModel) -> None:
    """Send with a timeout so a stalled client cannot pin the server (backpressure)."""
    await asyncio.wait_for(ws.send_text(msg.model_dump_json()), timeout=SEND_TIMEOUT_SECONDS)

"""Replay index and gzip passthrough (§27.3, §28.2)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from backend.app.api.deps import SafeId, SafeIdQuery, get_services
from backend.app.container import Services

router = APIRouter(tags=["replays"])


@router.get("/replays")
def list_replays(
    run_id: SafeIdQuery = None,
    map_id: SafeIdQuery = None,
    study: SafeIdQuery = None,
    algo: SafeIdQuery = None,
    svc: Services = Depends(get_services),
) -> list[dict[str, Any]]:
    return svc.replays.list(run_id, map_id, study, algo)


@router.get("/replays/{replay_id}/initial")
def get_replay_initial(replay_id: SafeId, svc: Services = Depends(get_services)) -> dict[str, Any]:
    """Cells known right after reset — not part of any frame delta (see `ReplayInitial`)."""
    return svc.replays.initial(replay_id)


@router.get("/replays/{replay_id}")
def get_replay(replay_id: SafeId, svc: Services = Depends(get_services)) -> FileResponse:
    """The stored `.json.gz` is served as-is (`Content-Encoding: gzip`); the
    browser decompresses it. Nothing is re-simulated or re-encoded."""
    path = svc.replays.path(replay_id)
    # `no-cache` = always revalidate; FileResponse supplies ETag/Last-Modified, so an
    # unchanged replay is a cheap 304 rather than a re-download.
    return FileResponse(
        path, media_type="application/json", headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"}
    )

"""Live simulation sessions (§28.2). Thin: validate, call the service, return."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from backend.app.api.deps import SafeId, get_services
from backend.app.container import Services
from backend.app.schemas.sessions import (
    SessionCreate,
    SessionReset,
    SessionSnapshot,
    StepRequest,
    StepResponse,
)
from backend.app.services.errors import ValidationFailed

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionSnapshot, status_code=201)
def create_session(req: SessionCreate, svc: Services = Depends(get_services)) -> SessionSnapshot:
    return svc.sessions.create(req).snapshot()


@router.get("/sessions/{session_id}", response_model=SessionSnapshot)
def get_session(session_id: SafeId, svc: Services = Depends(get_services)) -> SessionSnapshot:
    s = svc.sessions.get(session_id)
    with s.lock:
        return s.snapshot()


@router.post("/sessions/{session_id}/reset", response_model=SessionSnapshot)
def reset_session(
    session_id: SafeId, req: SessionReset | None = None, svc: Services = Depends(get_services)
) -> SessionSnapshot:
    s = svc.sessions.reset(session_id, req or SessionReset())
    with s.lock:
        return s.snapshot()


@router.post("/sessions/{session_id}/step", response_model=StepResponse)
def step_session(
    session_id: SafeId, req: StepRequest | None = None, svc: Services = Depends(get_services)
) -> StepResponse:
    n = (req or StepRequest()).n
    if n > svc.settings.max_steps_per_request:
        raise ValidationFailed(f"n must be between 1 and {svc.settings.max_steps_per_request}")
    return svc.sessions.get(session_id).step(n)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: SafeId, svc: Services = Depends(get_services)) -> Response:
    svc.sessions.close(session_id)
    return Response(status_code=204)

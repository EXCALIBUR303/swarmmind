"""GET /api/models, /api/models/{id} (§28.2)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.app.api.deps import SafeId, SafeIdQuery, get_services
from backend.app.container import Services

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models(algo: SafeIdQuery = None, level: SafeIdQuery = None, svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    return svc.models.list(algo, level)


@router.get("/models/{model_id}")
def get_model(model_id: SafeId, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return svc.models.get(model_id).info

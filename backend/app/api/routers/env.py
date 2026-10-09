"""GET /api/levels, GET /api/maps (§28.2)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import SafeIdQuery, get_services
from backend.app.container import Services

router = APIRouter(tags=["env"])


@router.get("/levels")
def levels(svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    return svc.maps.list_levels()


@router.get("/maps")
def maps(
    set: Annotated[Literal["validation", "test", "structured"], Query()] = "validation",
    level: SafeIdQuery = None,
    svc: Services = Depends(get_services),
) -> list[dict[str, Any]]:
    return svc.maps.list_maps(set, level)

"""Aggregated studies and precomputed analysis (§28.2)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import RequiredIdQuery, SafeId, get_services
from backend.app.container import Services

router = APIRouter(tags=["studies"])


@router.get("/studies")
def list_studies(svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    return svc.studies.list()


@router.get("/studies/{study_id}")
def get_study(study_id: SafeId, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return svc.studies.get(study_id)


@router.get("/analysis/{study_id}/heatmap")
def heatmap(
    study_id: SafeId,
    map_id: RequiredIdQuery,
    algo: RequiredIdQuery,
    level: RequiredIdQuery = "L2",
    kind: Annotated[Literal["visits", "collisions"], Query()] = "visits",
    svc: Services = Depends(get_services),
) -> dict[str, Any]:
    """Heatmaps were precomputed per (study, level, algo, map), aggregated over
    that study's seeds — not per run — so the spec's `/analysis/{run_id}/…`
    is keyed by study here."""
    return svc.studies.heatmap(study_id, level, algo, map_id, kind)

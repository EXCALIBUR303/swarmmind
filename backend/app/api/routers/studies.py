"""Aggregated studies and precomputed analysis (§28.2)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import RequiredIdQuery, SafeId, get_services
from backend.app.container import Services
from backend.app.services.study_service import preregistration_rules

router = APIRouter(tags=["studies"])


@router.get("/studies")
def list_studies(svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    return svc.studies.list()


@router.get("/preregistration")
def preregistration() -> dict[str, Any]:
    """Frozen decision rules (H1-H5) quoted verbatim from docs/EXPERIMENTS.md."""
    return preregistration_rules()


@router.get("/studies/{study_id}")
def get_study(study_id: SafeId, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return svc.studies.get(study_id)


@router.get("/studies/{study_id}/analysis/{name}")
def analysis_file(study_id: SafeId, name: SafeId, svc: Services = Depends(get_services)) -> Any:
    """A precomputed analysis JSON by file stem, e.g. `L2_coordination`, `L2_mappo_example_episode`
    (names are listed in `GET /studies/{id}` -> `analysis_files`)."""
    return svc.studies.analysis_file(study_id, name)


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

"""Run index, metrics and eval results (§28.2)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import SafeId, SafeIdQuery, get_services
from backend.app.container import Services

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
def list_runs(
    study: SafeIdQuery = None,
    algo: SafeIdQuery = None,
    level: SafeIdQuery = None,
    status: SafeIdQuery = None,
    svc: Services = Depends(get_services),
) -> list[dict[str, Any]]:
    return svc.runs.list_runs(study, algo, level, status)


@router.get("/experiments/{run_id}")
def get_run(run_id: SafeId, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return svc.runs.get(run_id)


@router.get("/experiments/{run_id}/metrics")
def run_metrics(
    run_id: SafeId, from_update: int = Query(0, ge=0), svc: Services = Depends(get_services)
) -> list[dict[str, Any]]:
    return svc.runs.metrics(run_id, from_update)


@router.get("/experiments/{run_id}/eval")
def run_eval(
    run_id: SafeId, episodes: bool = False, svc: Services = Depends(get_services)
) -> dict[str, Any]:
    return svc.runs.eval_results(run_id, include_episodes=episodes)

"""Training jobs (§28.2, §28.5). The API never trains in-process."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import SafeId, get_services
from backend.app.container import Services
from backend.app.schemas.training import JobStatus, TrainRequest

router = APIRouter(tags=["training"])


@router.post("/training/jobs", response_model=JobStatus, status_code=202)
def launch(req: TrainRequest, svc: Services = Depends(get_services)) -> JobStatus:
    return svc.jobs.submit(req)


@router.get("/training/jobs", response_model=list[JobStatus])
def list_jobs(svc: Services = Depends(get_services)) -> list[JobStatus]:
    return svc.jobs.list()


@router.get("/training/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: SafeId, svc: Services = Depends(get_services)) -> JobStatus:
    return svc.jobs.status(job_id)


@router.post("/training/jobs/{job_id}/cancel", response_model=JobStatus)
def cancel(job_id: SafeId, svc: Services = Depends(get_services)) -> JobStatus:
    return svc.jobs.cancel(job_id)

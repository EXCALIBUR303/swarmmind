"""GET /api/health — status, versions, device, run/model counts (§28.2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_services
from backend.app.container import Services
from marl.utils.versions import collect_versions

router = APIRouter(tags=["health"])


def _torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:  # pragma: no cover
        return "unknown"


@router.get("/health")
def health(svc: Services = Depends(get_services)) -> dict[str, object]:
    return {
        "status": "ok",
        "versions": collect_versions(),
        "device": _torch_device(),
        "counts": {
            "runs": len(svc.runs.list_runs()),
            "models": len(svc.models.list()),
            "replays": len(svc.replays.list()),
            "live_sessions": svc.sessions.count(),
        },
    }

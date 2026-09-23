"""GET /api/health — status, versions, device, run/model counts (§28.2)."""

from __future__ import annotations

from fastapi import APIRouter

from marl.utils.paths import MODELS_DIR, RUNS_DIR
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


def _count_dirs(path: object) -> int:
    from pathlib import Path

    p = Path(path)  # type: ignore[arg-type]
    if not p.is_dir():
        return 0
    return sum(1 for child in p.iterdir() if child.is_dir())


@router.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "versions": collect_versions(),
        "device": _torch_device(),
        "counts": {
            "runs": _count_dirs(RUNS_DIR),
            "models": _count_dirs(MODELS_DIR),
        },
    }

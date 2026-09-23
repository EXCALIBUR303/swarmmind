"""FastAPI app factory (§28.1).

Phase 1 scope: app factory, CORS, and the health router only. Sessions,
models, experiments, replays and training routers are added in Phase 10.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routers import health
from backend.app.settings import settings
from marl.utils.paths import ensure_dirs


def create_app() -> FastAPI:
    ensure_dirs()
    app = FastAPI(title="SWARMMIND API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")

    return app


app = create_app()

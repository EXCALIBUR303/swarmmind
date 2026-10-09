"""FastAPI app factory (§28.1): CORS, host allowlist, lifespan, routers."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.app.api.routers import (
    env,
    experiments,
    health,
    models,
    replays,
    sessions,
    studies,
    training,
)
from backend.app.api.ws import sim_ws, train_ws
from backend.app.container import Services
from backend.app.services.errors import ServiceError
from backend.app.settings import Settings
from backend.app.settings import settings as default_settings
from marl.utils.paths import ensure_dirs

REAP_INTERVAL_SECONDS = 30


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or default_settings
    ensure_dirs()
    services = Services.build(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(services.refresh)  # scan results/ and models/ once at startup

        async def reaper() -> None:
            while True:
                await asyncio.sleep(REAP_INTERVAL_SECONDS)
                services.sessions.reap()

        task = asyncio.create_task(reaper())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            services.sessions.close_all()

    app = FastAPI(title="SWARMMIND API", version="0.1.0", lifespan=lifespan)
    app.state.services = services

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[cfg.cors_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=cfg.allowed_hosts)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    for r in (health, env, sessions, models, experiments, studies, replays, training):
        app.include_router(r.router, prefix="/api")
    app.include_router(sim_ws.router)
    app.include_router(train_ws.router)
    return app


app = create_app()

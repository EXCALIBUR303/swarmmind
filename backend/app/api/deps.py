"""Shared router dependencies and validated path/query types (§37)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Path, Query, Request

from backend.app.container import Services

# First char may not be '.', so '.', '..' and hidden names never validate.
ID_RE = r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$"

SafeId = Annotated[str, Path(pattern=ID_RE)]
SafeIdQuery = Annotated[str | None, Query(pattern=ID_RE)]
RequiredIdQuery = Annotated[str, Query(pattern=ID_RE)]


def get_services(request: Request) -> Services:
    return request.app.state.services

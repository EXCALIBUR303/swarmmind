"""Service-layer exceptions; routers map these to HTTP status codes."""

from __future__ import annotations


class ServiceError(Exception):
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(ServiceError):
    status_code = 404


class ConflictError(ServiceError):
    status_code = 409


class ValidationFailed(ServiceError):
    status_code = 422


class CapacityError(ServiceError):
    status_code = 429

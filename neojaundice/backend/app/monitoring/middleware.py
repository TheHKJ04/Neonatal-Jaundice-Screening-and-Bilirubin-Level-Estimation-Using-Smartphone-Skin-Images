"""ASGI middleware for HTTP metrics and structured access logs."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from backend.app.monitoring.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS,
)

logger = logging.getLogger("neojaundice.access")

# Only these routes can appear as metric labels.
# Unknown paths are grouped under "other".
KNOWN_ROUTES = {
    "/api/v1/predict",
    "/api/v1/health",
    "/api/v1/models/status",
    "/health",
    "/models/status",
}


class ObservabilityMiddleware:
    """Collect HTTP request metrics without recording sensitive payloads."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")

        # Do not count Prometheus scraping as ordinary application traffic.
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        route = path if path in KNOWN_ROUTES else "other"

        started_at = perf_counter()
        status_code = 500
        exception_type: str | None = None

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            exception_type = type(exc).__name__
            raise
        finally:
            duration_seconds = perf_counter() - started_at

            HTTP_REQUESTS.labels(
                method=method,
                route=route,
                status_code=str(status_code),
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                route=route,
            ).observe(duration_seconds)

            log_fields = {
                "event": "http_request",
                "method": method,
                "route": route,
                "status_code": status_code,
                "duration_ms": round(duration_seconds * 1000, 3),
            }

            if exception_type is not None:
                log_fields["exception_type"] = exception_type

            logger.info(
                "HTTP request completed",
                extra=log_fields,
            )

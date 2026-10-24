"""Prometheus metrics endpoint."""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["monitoring"])


@router.get(
    "/metrics",
    include_in_schema=False,
    response_class=Response,
)
def metrics() -> Response:
    """Return metrics in Prometheus exposition format."""

    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )

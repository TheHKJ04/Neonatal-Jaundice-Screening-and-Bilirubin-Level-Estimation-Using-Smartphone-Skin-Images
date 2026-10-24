"""Central definitions for NeoJaundice Prometheus metrics."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from prometheus_client import Counter, Histogram

# HTTP_REQUESTS tracks completed HTTP requests.
# Labels are deliberately restricted to prevent high-cardinality metrics.
HTTP_REQUESTS = Counter(
    "neojaundice_http_requests",
    "Total number of completed HTTP requests.",
    labelnames=("method", "route", "status_code"),
)

# HTTP_REQUEST_DURATION_SECONDS measures total HTTP request duration.
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "neojaundice_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    labelnames=("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# MODEL_INFERENCE_DURATION_SECONDS measures time spent inside the
# model-registry prediction call, separately from HTTP processing.
MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    "neojaundice_model_inference_duration_seconds",
    "Duration of the model-registry inference call in seconds.",
    labelnames=("outcome",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODEL_INFERENCE_ERRORS = Counter(
    "neojaundice_model_inference_errors",
    "Number of model-registry inference calls that raised exceptions.",
)


@contextmanager
def track_model_inference() -> Iterator[None]:
    """Record inference duration and whether the call raised an exception."""

    started_at = perf_counter()
    outcome = "error"

    try:
        yield
    except Exception:
        MODEL_INFERENCE_ERRORS.inc()
        raise
    else:
        outcome = "success"
    finally:
        elapsed_seconds = perf_counter() - started_at
        MODEL_INFERENCE_DURATION_SECONDS.labels(outcome=outcome).observe(
            elapsed_seconds
        )

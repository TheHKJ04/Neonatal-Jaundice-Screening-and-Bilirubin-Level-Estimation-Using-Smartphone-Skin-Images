"""Tests for NeoJaundice monitoring metrics."""

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.monitoring.metrics import track_model_inference


def test_metrics_endpoint_returns_prometheus_metrics() -> None:
    client = TestClient(app)

    # Generate at least one completed application request.
    health_response = client.get("/api/v1/health")
    assert health_response.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "neojaundice_http_requests_total" in response.text
    assert "neojaundice_http_request_duration_seconds" in response.text


def test_model_inference_metrics_record_success() -> None:
    client = TestClient(app)

    with track_model_inference():
        pass

    response = client.get("/metrics")

    assert response.status_code == 200
    assert (
        'neojaundice_model_inference_duration_seconds_count{outcome="success"}'
        in response.text
    )


def test_model_inference_metrics_record_failure() -> None:
    client = TestClient(app)

    try:
        with track_model_inference():
            raise RuntimeError("Intentional test exception")
    except RuntimeError as exc:
        assert str(exc) == "Intentional test exception"
    else:
        raise AssertionError("Expected RuntimeError was not raised")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert (
        'neojaundice_model_inference_duration_seconds_count{outcome="error"}'
        in response.text
    )
    assert "neojaundice_model_inference_errors_total" in response.text

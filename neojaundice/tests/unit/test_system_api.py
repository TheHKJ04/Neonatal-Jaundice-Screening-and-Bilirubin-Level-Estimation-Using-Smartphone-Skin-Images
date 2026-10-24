"""API behavior and isolated model inventory tests."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_health(tmp_path: Path) -> None:
    settings = Settings(
        app_name="Test Research API",
        regression_checkpoint=tmp_path / "regression.pt",
        classification_checkpoint=tmp_path / "classification.pt",
    )
    client = TestClient(create_app(settings))
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Test Research API"}


def test_missing_models_are_not_reported_as_loaded(tmp_path: Path) -> None:
    settings = Settings(
        regression_checkpoint=tmp_path / "missing-regression.pt",
        classification_checkpoint=tmp_path / "missing-classification.pt",
    )
    response = TestClient(create_app(settings)).get("/api/v1/models/status")
    assert response.status_code == 200
    assert response.json() == {
        "regression": {"available": False, "loaded": False},
        "classification": {"available": False, "loaded": False},
        "research_only": True,
    }


def test_checkpoint_presence_is_not_model_loading(tmp_path: Path) -> None:
    regression = tmp_path / "regression.pt"
    regression.write_bytes(b"placeholder")
    settings = Settings(
        regression_checkpoint=regression,
        classification_checkpoint=tmp_path / "missing.pt",
    )
    response = TestClient(create_app(settings)).get("/api/v1/models/status")
    assert response.json()["regression"] == {"available": True, "loaded": False}
    assert response.json()["classification"] == {"available": False, "loaded": False}


def test_openapi_is_available(tmp_path: Path) -> None:
    settings = Settings(
        regression_checkpoint=tmp_path / "r.pt",
        classification_checkpoint=tmp_path / "c.pt",
    )
    response = TestClient(create_app(settings)).get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]

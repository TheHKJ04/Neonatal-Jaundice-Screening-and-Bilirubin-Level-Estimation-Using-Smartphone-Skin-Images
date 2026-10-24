"""Tests for the NeoJaundice image prediction API."""

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import Settings
from backend.app.main import create_app


def create_test_image() -> bytes:
    """Create a small valid PNG image for API tests."""

    image = Image.new(
        mode="RGB",
        size=(32, 32),
        color=(180, 150, 120),
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def create_test_app(
    tmp_path: Path,
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> TestClient:
    """Create an isolated application with temporary checkpoint paths."""

    settings = Settings(
        app_env="testing",
        regression_checkpoint=tmp_path / "regression.pt",
        classification_checkpoint=tmp_path / "classification.pt",
        max_upload_bytes=max_upload_bytes,
    )

    app = create_app(settings)

    # API contract test double: no trained model is needed for these tests.
    app.state.model_registry.predict = lambda image_tensor: {
        "regression_prediction": 7.25,
        "classification_probability": 0.82,
        "classification_prediction": 1,
    }

    return TestClient(app)


def test_prediction_accepts_valid_image(tmp_path: Path) -> None:
    """A valid image should produce the documented response schema."""

    client = create_test_app(tmp_path)

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "skin.png",
                create_test_image(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["regression_prediction"] == 7.25
    assert body["classification_probability"] == 0.82
    assert body["classification_prediction"] == 1
    assert body["research_only"] is True
    assert "not intended for clinical diagnosis" in body["message"].lower()


def test_prediction_rejects_empty_file(tmp_path: Path) -> None:
    """An empty upload should return HTTP 400."""

    client = create_test_app(tmp_path)

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded image is empty."


def test_prediction_rejects_corrupt_image(tmp_path: Path) -> None:
    """A non-image payload should return HTTP 400."""

    client = create_test_app(tmp_path)

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "corrupt.png",
                b"this is not a real image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_prediction_rejects_unsupported_image_format(
    tmp_path: Path,
) -> None:
    """A valid BMP image should be rejected by the format allowlist."""

    client = create_test_app(tmp_path)

    image = Image.new("RGB", (32, 32))
    buffer = BytesIO()
    image.save(buffer, format="BMP")

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "skin.bmp",
                buffer.getvalue(),
                "image/bmp",
            )
        },
    )

    assert response.status_code == 415


def test_prediction_rejects_oversized_upload(tmp_path: Path) -> None:
    """Uploads above the configured limit should return HTTP 413."""

    client = create_test_app(
        tmp_path,
        max_upload_bytes=1024,
    )

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "large.png",
                b"x" * 1025,
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_prediction_rejects_missing_model_checkpoint(
    tmp_path: Path,
) -> None:
    """Real registry inference should fail safely if checkpoints are missing."""

    settings = Settings(
        app_env="testing",
        regression_checkpoint=tmp_path / "missing-regression.pt",
        classification_checkpoint=tmp_path / "missing-classification.pt",
    )

    client = TestClient(create_app(settings))

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "skin.png",
                create_test_image(),
                "image/png",
            )
        },
    )

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]

"""Integration tests using the real NeoJaundice checkpoints.

These tests are skipped when the configured checkpoint files are absent.
They verify software integration, not clinical performance.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image, ImageOps

from backend.app.core.config import get_settings
from backend.app.services.model_registry import ModelRegistry
from ml.training.dataset import make_transform


@pytest.fixture(scope="module")
def model_registry() -> ModelRegistry:
    """Create a registry using the configured local checkpoints."""

    settings = get_settings()

    checkpoint_paths = (
        Path(settings.regression_checkpoint),
        Path(settings.classification_checkpoint),
    )

    missing = [str(path) for path in checkpoint_paths if not path.is_file()]

    if missing:
        pytest.skip(
            "Real-model integration test requires both checkpoints. "
            f"Missing: {', '.join(missing)}"
        )

    return ModelRegistry(
        regression_checkpoint=checkpoint_paths[0],
        classification_checkpoint=checkpoint_paths[1],
    )


@pytest.fixture(scope="module")
def synthetic_image_tensor() -> torch.Tensor:
    """Create a synthetic image to exercise preprocessing and inference.

    This is a software test image, not a neonatal clinical photograph.
    """

    image = Image.new("RGB", (320, 240), color=(180, 120, 100))

    image = ImageOps.exif_transpose(image).convert("RGB")

    transform = make_transform(
        mode="resize",
        training=False,
    )

    return transform(image).unsqueeze(0)


def test_real_checkpoints_load_and_produce_valid_outputs(
    model_registry: ModelRegistry,
    synthetic_image_tensor: torch.Tensor,
) -> None:
    """Verify both trained models can run through the registry."""

    with torch.inference_mode():
        result = model_registry.predict(synthetic_image_tensor)

    assert set(result) == {
        "regression_prediction",
        "classification_probability",
        "classification_prediction",
    }

    regression_value = result["regression_prediction"]
    probability = result["classification_probability"]
    predicted_class = result["classification_prediction"]

    assert isinstance(regression_value, float)
    assert torch.isfinite(torch.tensor(regression_value)).item()

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0

    assert predicted_class in (0, 1)
    assert predicted_class == int(probability >= 0.5)

    status = model_registry.status()

    assert status.regression.available is True
    assert status.classification.available is True

    assert status.regression.loaded is True
    assert status.classification.loaded is True


def test_repeated_inference_returns_consistent_results(
    model_registry: ModelRegistry,
    synthetic_image_tensor: torch.Tensor,
) -> None:
    """Check deterministic behavior for repeated inference on the same input."""

    first = model_registry.predict(synthetic_image_tensor)
    second = model_registry.predict(synthetic_image_tensor)

    assert first["regression_prediction"] == pytest.approx(
        second["regression_prediction"],
        abs=1e-5,
    )

    assert first["classification_probability"] == pytest.approx(
        second["classification_probability"],
        abs=1e-5,
    )

    assert first["classification_prediction"] == second["classification_prediction"]

"""Unit tests for the NeoJaundice model registry."""

from pathlib import Path

import pytest
import torch
from torch import nn

from backend.app.services import model_registry as registry_module
from backend.app.services.model_registry import (
    ModelRegistry,
    ModelUnavailableError,
)


class FakeModel(nn.Module):
    """Tiny model that returns one constant logit per input image."""

    def __init__(self) -> None:
        super().__init__()
        self.output_value = nn.Parameter(torch.tensor([0.0]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_value.unsqueeze(0).expand(inputs.shape[0], 1)


def create_registry(tmp_path: Path) -> ModelRegistry:
    regression_path = tmp_path / "regression.pt"
    classification_path = tmp_path / "classification.pt"

    # Empty placeholder files are sufficient because torch.load is mocked.
    regression_path.touch()
    classification_path.touch()

    return ModelRegistry(
        regression_checkpoint=regression_path,
        classification_checkpoint=classification_path,
    )


def test_status_distinguishes_available_from_loaded(tmp_path: Path) -> None:
    registry = create_registry(tmp_path)

    status = registry.status()

    assert status.regression.available is True
    assert status.classification.available is True
    assert status.regression.loaded is False
    assert status.classification.loaded is False


def test_missing_checkpoint_raises_model_unavailable_error(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(
        regression_checkpoint=tmp_path / "missing-regression.pt",
        classification_checkpoint=tmp_path / "missing-classification.pt",
    )

    with pytest.raises(ModelUnavailableError, match="checkpoint is unavailable"):
        registry._get_regression_model()


def test_predict_returns_expected_fields_and_caches_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = create_registry(tmp_path)

    model_build_count = 0

    def fake_build_model(*, pretrained: bool) -> FakeModel:
        nonlocal model_build_count
        assert pretrained is False
        model_build_count += 1
        return FakeModel()

    def fake_torch_load(
        checkpoint_path: Path,
        *,
        map_location: str,
        weights_only: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        assert map_location == "cpu"
        assert weights_only is True
        return {"model_state_dict": {"output_value": torch.tensor([0.0])}}

    monkeypatch.setattr(registry_module, "build_model", fake_build_model)
    monkeypatch.setattr(torch, "load", fake_torch_load)

    image_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    first_result = registry.predict(image_tensor)
    second_result = registry.predict(image_tensor)

    assert set(first_result) == {
        "regression_prediction",
        "classification_probability",
        "classification_prediction",
    }

    assert first_result["regression_prediction"] == pytest.approx(0.0)
    assert first_result["classification_probability"] == pytest.approx(0.5)
    assert first_result["classification_prediction"] == 1

    assert second_result == first_result

    # Exactly two model instances are built: one regression and one classifier.
    # They are reused for the second prediction.
    assert model_build_count == 2

    status = registry.status()
    assert status.regression.loaded is True
    assert status.classification.loaded is True


@pytest.mark.parametrize(
    "invalid_tensor",
    [
        torch.zeros((3, 224, 224)),
        torch.zeros((1, 1, 224, 224)),
        torch.zeros((2, 3, 224, 224)),
        torch.zeros((1, 3, 224, 224), dtype=torch.int64),
    ],
)
def test_predict_rejects_invalid_input_tensor(
    tmp_path: Path,
    invalid_tensor: torch.Tensor,
) -> None:
    registry = create_registry(tmp_path)

    with pytest.raises(ValueError):
        registry.predict(invalid_tensor)


def test_predict_rejects_non_finite_input(
    tmp_path: Path,
) -> None:
    registry = create_registry(tmp_path)

    invalid_tensor = torch.zeros((1, 3, 224, 224))
    invalid_tensor[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        registry.predict(invalid_tensor)

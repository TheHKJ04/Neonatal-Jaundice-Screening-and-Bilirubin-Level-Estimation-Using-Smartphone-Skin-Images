"""Safe, lazy-loading registry for NeoJaundice research models."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from backend.app.schemas.system import ModelsResponse, ModelStatus
from ml.training.model import build_model

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when a required model cannot be loaded or used."""


class ModelRegistry:
    """Manage checkpoint availability and cached inference models.

    Models are loaded only when prediction is requested. This avoids
    loading large checkpoints when the application starts or when only
    the health endpoint is used.
    """

    def __init__(
        self,
        regression_checkpoint: Path,
        classification_checkpoint: Path,
    ) -> None:
        self.regression_checkpoint = Path(regression_checkpoint)
        self.classification_checkpoint = Path(classification_checkpoint)

        self._regression_model: nn.Module | None = None
        self._classification_model: nn.Module | None = None

        # Prevent concurrent requests from loading the same model twice.
        self._lock = threading.RLock()

        # Inference initially runs on CPU for predictable local deployment.
        self._device = torch.device("cpu")

    def status(self) -> ModelsResponse:
        """Report checkpoint availability separately from model loading."""
        with self._lock:
            return ModelsResponse(
                regression=ModelStatus(
                    available=self.regression_checkpoint.is_file(),
                    loaded=self._regression_model is not None,
                ),
                classification=ModelStatus(
                    available=self.classification_checkpoint.is_file(),
                    loaded=self._classification_model is not None,
                ),
            )

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
        """Extract a tensor state dictionary from a supported checkpoint."""

        if not isinstance(checkpoint, Mapping):
            raise ModelUnavailableError(
                "Checkpoint must contain a model state dictionary."
            )

        # Support common checkpoint formats used by PyTorch training scripts.
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)

            if isinstance(candidate, Mapping):
                if candidate and all(
                    isinstance(value, torch.Tensor) for value in candidate.values()
                ):
                    return candidate

        # Also support checkpoints containing the state dictionary directly.
        if checkpoint and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            return checkpoint

        raise ModelUnavailableError(
            "No valid tensor state dictionary was found in the checkpoint."
        )

    def _load_model(self, checkpoint_path: Path) -> nn.Module:
        """Load one EfficientNet model and validate its architecture."""

        if not checkpoint_path.is_file():
            raise ModelUnavailableError(
                f"Required model checkpoint is unavailable: {checkpoint_path.name}"
            )

        try:
            # weights_only=True limits deserialization to supported safe
            # objects. Do not fall back to unrestricted pickle loading.
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            state_dict = self._extract_state_dict(checkpoint)

            model = build_model(pretrained=False)
            model.load_state_dict(state_dict, strict=True)
            model.to(self._device)
            model.eval()

            logger.info(
                "Loaded model checkpoint: %s",
                checkpoint_path.name,
            )
            return model

        except ModelUnavailableError:
            raise
        except Exception as exc:
            logger.exception(
                "Could not load model checkpoint: %s",
                checkpoint_path.name,
            )
            raise ModelUnavailableError(
                f"Model checkpoint could not be loaded: {checkpoint_path.name}"
            ) from exc

    def _get_regression_model(self) -> nn.Module:
        """Return the cached regression model, loading it if necessary."""
        with self._lock:
            if self._regression_model is None:
                self._regression_model = self._load_model(self.regression_checkpoint)
            return self._regression_model

    def _get_classification_model(self) -> nn.Module:
        """Return the cached classification model, loading it if necessary."""
        with self._lock:
            if self._classification_model is None:
                self._classification_model = self._load_model(
                    self.classification_checkpoint
                )
            return self._classification_model

    def predict(self, image_tensor: torch.Tensor) -> dict[str, float | int]:
        """Run regression and classification inference for one image.

        Expected input shape: [1, 3, 224, 224].
        The input must already use the training-time image preprocessing.
        """

        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")

        if tuple(image_tensor.shape) != (1, 3, 224, 224):
            raise ValueError("Expected an image tensor with shape [1, 3, 224, 224].")

        if not torch.is_floating_point(image_tensor):
            raise ValueError("Input image tensor must use a floating-point dtype.")

        if not torch.isfinite(image_tensor).all().item():
            raise ValueError("Input image tensor contains non-finite values.")

        try:
            regression_model = self._get_regression_model()
            classification_model = self._get_classification_model()

            image_tensor = image_tensor.to(self._device)

            with torch.inference_mode():
                regression_output = regression_model(image_tensor)
                classification_logits = classification_model(image_tensor)

                if regression_output.numel() != 1:
                    raise ModelUnavailableError(
                        "Regression model must return one value per image."
                    )

                if classification_logits.numel() != 1:
                    raise ModelUnavailableError(
                        "Classification model must return one logit per image."
                    )

                regression_value = float(regression_output.reshape(-1)[0].item())

                probability = float(
                    torch.sigmoid(classification_logits.reshape(-1)[0]).item()
                )

            return {
                "regression_prediction": regression_value,
                "classification_probability": probability,
                "classification_prediction": int(probability >= 0.5),
            }

        except ModelUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Model inference failed.")
            raise ModelUnavailableError(
                "Model inference could not be completed."
            ) from exc

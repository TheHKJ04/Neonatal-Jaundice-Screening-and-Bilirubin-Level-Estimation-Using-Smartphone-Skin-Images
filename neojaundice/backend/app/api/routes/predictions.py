"""Image prediction endpoint for the NeoJaundice research API."""

import logging
from io import BytesIO

import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.app.monitoring.metrics import track_model_inference
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.model_registry import ModelUnavailableError
from ml.training.dataset import make_transform

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["predictions"],
)

MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "JPG"}

# Reuse the same deterministic preprocessing used during model evaluation.
INFERENCE_TRANSFORM = make_transform(
    mode="resize",
    training=False,
)


def preprocess_uploaded_image(image_bytes: bytes) -> torch.Tensor:
    """Decode, validate, and preprocess an uploaded image."""

    try:
        # First inspect the file format and dimensions.
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format. Use JPEG, PNG, or WebP.",
                )

            width, height = image.size

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are invalid.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the permitted limit.",
                )

            # Check image integrity before decoding it for inference.
            image.verify()

        # Reopen the image after verify(), correct EXIF orientation,
        # and convert it to the RGB format expected by the model.
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            # Resize, convert to tensor, and apply ImageNet normalization.
            tensor = INFERENCE_TRANSFORM(image).unsqueeze(0)

        return tensor

    except HTTPException:
        raise

    except Image.DecompressionBombError as exc:
        raise HTTPException(
            status_code=413,
            detail="Image dimensions exceed the permitted limit.",
        ) from exc

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        ) from exc


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict from a skin image",
    description=(
        "Research-only inference using the regression and treatment-label "
        "classification models. This endpoint does not provide a diagnosis "
        "or treatment recommendation."
    ),
)
async def predict_image(
    request: Request,
    file: UploadFile = File(
        ...,
        description="JPEG, PNG, or WebP image",
    ),
) -> PredictionResponse:
    """Validate an uploaded image and return both model outputs."""

    max_upload_bytes = request.app.state.settings.max_upload_bytes

    try:
        # Read at most one byte beyond the limit so oversized uploads
        # can be detected without reading the entire file into memory.
        image_bytes = await file.read(max_upload_bytes + 1)
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="The uploaded file exceeds the permitted size limit.",
        )

    # Input validation errors are handled separately from inference errors.
    image_tensor = preprocess_uploaded_image(image_bytes)

    try:
        with track_model_inference():
            predictions = request.app.state.model_registry.predict(image_tensor)
        return PredictionResponse(**predictions)

    except ModelUnavailableError as exc:
        logger.warning(
            "Prediction unavailable because a required model could not load."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction is temporarily unavailable because "
                "a required model is not ready."
            ),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during image inference.")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction.",
        ) from exc

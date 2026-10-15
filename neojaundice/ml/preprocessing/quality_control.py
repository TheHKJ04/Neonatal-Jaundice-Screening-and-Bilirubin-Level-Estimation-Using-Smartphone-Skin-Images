"""Quality validation for NeoJaundice preprocessing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ml.preprocessing.image_preprocessor import (
    PreprocessingConfig,
    PreprocessingResult,
)


@dataclass(frozen=True)
class QCResult:
    """Validation result for one preprocessed image."""

    passed: bool
    errors: tuple[str, ...]
    valid_pixel_fraction: float
    image_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]


def validate_preprocessing_result(
    result: PreprocessingResult,
    config: PreprocessingConfig,
) -> QCResult:
    """Validate tensor, mask, geometry, and preview consistency."""

    errors: list[str] = []

    expected_image_shape = (
        3,
        config.target_height,
        config.target_width,
    )

    expected_mask_shape = (
        1,
        config.target_height,
        config.target_width,
    )

    image = result.image_tensor
    mask = result.valid_mask
    preview = result.preview_rgb
    metadata = result.metadata

    if tuple(image.shape) != expected_image_shape:
        errors.append("invalid_image_shape")

    if tuple(mask.shape) != expected_mask_shape:
        errors.append("invalid_mask_shape")

    if image.dtype != torch.float32:
        errors.append("invalid_image_dtype")

    if mask.dtype != torch.float32:
        errors.append("invalid_mask_dtype")

    if not torch.isfinite(image).all().item():
        errors.append("non_finite_image_values")

    if not torch.isfinite(mask).all().item():
        errors.append("non_finite_mask_values")

    if (
        not torch.logical_or(
            mask == 0,
            mask == 1,
        )
        .all()
        .item()
    ):
        errors.append("non_binary_mask")

    expected_preview_shape = (
        config.target_height,
        config.target_width,
        3,
    )

    if preview.shape != expected_preview_shape:
        errors.append("invalid_preview_shape")

    if preview.dtype != np.uint8:
        errors.append("invalid_preview_dtype")

    geometry_values = (
        metadata.padding_left,
        metadata.padding_right,
        metadata.padding_top,
        metadata.padding_bottom,
        metadata.resized_width,
        metadata.resized_height,
    )

    if any(value < 0 for value in geometry_values):
        errors.append("negative_geometry_value")

    if (
        metadata.resized_width + metadata.padding_left + metadata.padding_right
        != config.target_width
    ):
        errors.append("invalid_horizontal_geometry")

    if (
        metadata.resized_height + metadata.padding_top + metadata.padding_bottom
        != config.target_height
    ):
        errors.append("invalid_vertical_geometry")

    if (
        metadata.target_width != config.target_width
        or metadata.target_height != config.target_height
    ):
        errors.append("metadata_target_mismatch")

    if metadata.resized_width <= 0 or metadata.resized_height <= 0:
        errors.append("invalid_resized_dimensions")

    if (
        metadata.original_width <= 0
        or metadata.original_height <= 0
        or metadata.oriented_width <= 0
        or metadata.oriented_height <= 0
    ):
        errors.append("invalid_original_dimensions")

    valid_fraction = float("nan")

    if tuple(mask.shape) == expected_mask_shape:
        valid_fraction = float(mask.mean().item())

        expected_mask = torch.zeros_like(mask)

        left = metadata.padding_left
        top = metadata.padding_top

        right = left + metadata.resized_width
        bottom = top + metadata.resized_height

        if (
            0 <= left < right <= config.target_width
            and 0 <= top < bottom <= config.target_height
        ):
            expected_mask[
                :,
                top:bottom,
                left:right,
            ] = 1.0

            if not torch.equal(
                mask,
                expected_mask,
            ):
                errors.append("mask_geometry_mismatch")
        else:
            errors.append("mask_bounds_invalid")

    if (
        tuple(image.shape) == expected_image_shape
        and preview.shape == expected_preview_shape
        and preview.dtype == np.uint8
        and image.dtype == torch.float32
        and torch.isfinite(image).all().item()
    ):
        preview_tensor = (
            torch.from_numpy(preview.astype(np.float32).transpose(2, 0, 1).copy())
            / 255.0
        )

        if config.normalization == "imagenet":
            mean = torch.tensor(
                (0.485, 0.456, 0.406),
                dtype=torch.float32,
            ).view(3, 1, 1)

            std = torch.tensor(
                (0.229, 0.224, 0.225),
                dtype=torch.float32,
            ).view(3, 1, 1)

            preview_tensor = (preview_tensor - mean) / std

        if not torch.allclose(
            image,
            preview_tensor,
            atol=1e-5,
            rtol=1e-5,
        ):
            errors.append("tensor_preview_mismatch")

    return QCResult(
        passed=not errors,
        errors=tuple(errors),
        valid_pixel_fraction=valid_fraction,
        image_shape=tuple(image.shape),
        mask_shape=tuple(mask.shape),
    )


def summarize_qc_results(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a machine-readable QC summary."""

    total = len(records)

    passed = sum(record["passed"] is True for record in records)

    failed = total - passed

    region_counts: dict[str, int] = {}

    for record in records:
        region = str(record.get("body_region", "unknown"))

        region_counts[region] = region_counts.get(region, 0) + 1

    return {
        "total_images": total,
        "passed_images": passed,
        "failed_images": failed,
        "pass_rate": (passed / total if total else 0.0),
        "region_counts": region_counts,
        "status": ("passed" if total > 0 and failed == 0 else "failed"),
    }

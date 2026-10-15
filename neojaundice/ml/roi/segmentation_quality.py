"""Technical validation for candidate skin segmentation masks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ml.roi.skin_segmentation import SegmentationResult


@dataclass(frozen=True)
class SegmentationQualityResult:
    """Validation outcome for a candidate segmentation mask."""

    passed: bool
    errors: tuple[str, ...]
    selected_pixel_count: int
    coverage: float


def validate_segmentation_result(
    result: SegmentationResult,
    geometric_roi_mask: torch.Tensor,
) -> SegmentationQualityResult:
    """Validate segmentation integrity against the geometric ROI."""

    errors: list[str] = []

    mask = result.mask

    expected_shape = tuple(geometric_roi_mask.shape)

    shape_ok = (
        tuple(mask.shape) == expected_shape
        and len(expected_shape) == 3
        and expected_shape[0] == 1
    )

    if not shape_ok:
        errors.append("invalid_mask_shape")

    if mask.dtype != torch.float32:
        errors.append("invalid_mask_dtype")

    mask_finite = bool(torch.isfinite(mask).all().item())

    roi_finite = bool(torch.isfinite(geometric_roi_mask).all().item())

    if not mask_finite:
        errors.append("non_finite_mask")

    if not roi_finite:
        errors.append("non_finite_geometric_roi")

    mask_binary = bool(
        torch.logical_or(
            mask == 0,
            mask == 1,
        )
        .all()
        .item()
    )

    roi_binary = bool(
        torch.logical_or(
            geometric_roi_mask == 0,
            geometric_roi_mask == 1,
        )
        .all()
        .item()
    )

    if not mask_binary:
        errors.append("non_binary_mask")

    if not roi_binary:
        errors.append("non_binary_geometric_roi")

    selected_pixels = 0
    geometric_pixels = 0

    if mask_finite and mask_binary:
        selected_pixels = int(mask.sum().item())

    if roi_finite and roi_binary:
        geometric_pixels = int(geometric_roi_mask.sum().item())

    if geometric_pixels == 0:
        errors.append("empty_geometric_roi")

    if shape_ok and mask_finite and roi_finite and mask_binary and roi_binary:
        if not torch.all(mask <= geometric_roi_mask).item():
            errors.append("mask_outside_geometric_roi")

    if selected_pixels != result.selected_pixel_count:
        errors.append("selected_pixel_count_mismatch")

    if geometric_pixels != result.geometric_roi_pixel_count:
        errors.append("geometric_pixel_count_mismatch")

    coverage = selected_pixels / geometric_pixels if geometric_pixels > 0 else 0.0

    if not math.isclose(
        coverage,
        result.coverage,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        errors.append("coverage_mismatch")

    return SegmentationQualityResult(
        passed=len(errors) == 0,
        errors=tuple(errors),
        selected_pixel_count=selected_pixels,
        coverage=coverage,
    )

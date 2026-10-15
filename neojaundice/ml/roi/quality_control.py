"""Technical quality control for geometric ROI extraction."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch

from ml.preprocessing.image_preprocessor import (
    PreprocessingResult,
)
from ml.roi.geometric_roi import ROIResult


@dataclass(frozen=True)
class ROIQualityResult:
    """Technical validation outcome for one ROI."""

    passed: bool
    errors: tuple[str, ...]
    roi_pixel_count: int
    valid_pixel_count: int
    roi_fraction_of_valid_image: float


def validate_roi_result(
    preprocessing_result: PreprocessingResult,
    roi_result: ROIResult,
) -> ROIQualityResult:
    """Independently validate an ROI mask and its metadata."""

    errors: list[str] = []

    valid_mask = preprocessing_result.valid_mask
    roi_mask = roi_result.roi_mask

    metadata = preprocessing_result.metadata

    expected_shape = (
        1,
        metadata.target_height,
        metadata.target_width,
    )

    valid_shape_ok = tuple(valid_mask.shape) == expected_shape
    roi_shape_ok = tuple(roi_mask.shape) == expected_shape

    if not valid_shape_ok:
        errors.append("invalid_valid_mask_shape")

    if not roi_shape_ok:
        errors.append("invalid_roi_mask_shape")

    if roi_mask.dtype != torch.float32:
        errors.append("invalid_roi_mask_dtype")

    if valid_mask.dtype != torch.float32:
        errors.append("invalid_valid_mask_dtype")

    roi_finite = bool(torch.isfinite(roi_mask).all().item())
    valid_finite = bool(torch.isfinite(valid_mask).all().item())

    if not roi_finite:
        errors.append("non_finite_roi_mask")

    if not valid_finite:
        errors.append("non_finite_valid_mask")

    roi_binary = bool(
        torch.logical_or(
            roi_mask == 0,
            roi_mask == 1,
        )
        .all()
        .item()
    )

    valid_binary = bool(
        torch.logical_or(
            valid_mask == 0,
            valid_mask == 1,
        )
        .all()
        .item()
    )

    if not roi_binary:
        errors.append("non_binary_roi_mask")

    if not valid_binary:
        errors.append("non_binary_valid_mask")

    roi_pixels = 0
    valid_pixels = 0

    if roi_shape_ok and roi_finite and roi_binary:
        roi_pixels = int(roi_mask.sum().item())

    if valid_shape_ok and valid_finite and valid_binary:
        valid_pixels = int(valid_mask.sum().item())

    if roi_pixels == 0:
        errors.append("empty_roi")

    if valid_pixels == 0:
        errors.append("empty_valid_image")

    if (
        roi_shape_ok
        and valid_shape_ok
        and roi_finite
        and valid_finite
        and roi_binary
        and valid_binary
    ):
        if not torch.all(roi_mask <= valid_mask).item():
            errors.append("roi_includes_padding")

    if roi_pixels != roi_result.roi_pixel_count:
        errors.append("roi_pixel_count_mismatch")

    if valid_pixels != roi_result.valid_pixel_count:
        errors.append("valid_pixel_count_mismatch")

    x_min, y_min, x_max, y_max = roi_result.pixel_box

    box_valid = (
        0 <= x_min < x_max <= metadata.target_width
        and 0 <= y_min < y_max <= metadata.target_height
    )

    if not box_valid:
        errors.append("invalid_roi_box")

    if (
        box_valid
        and roi_shape_ok
        and valid_shape_ok
        and roi_finite
        and valid_finite
        and roi_binary
        and valid_binary
    ):
        expected_mask = torch.zeros_like(
            valid_mask,
            dtype=torch.float32,
        )

        expected_mask[
            :,
            y_min:y_max,
            x_min:x_max,
        ] = 1.0

        expected_mask = expected_mask * valid_mask

        if not torch.equal(
            roi_mask,
            expected_mask,
        ):
            errors.append("roi_geometry_mismatch")

    fraction = roi_pixels / valid_pixels if valid_pixels > 0 else 0.0

    if not math.isclose(
        fraction,
        roi_result.roi_fraction_of_valid_image,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        errors.append("roi_coverage_mismatch")

    canvas_pixels = metadata.target_width * metadata.target_height

    canvas_fraction = roi_pixels / canvas_pixels

    if not math.isclose(
        canvas_fraction,
        roi_result.roi_fraction_of_canvas,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        errors.append("canvas_coverage_mismatch")

    return ROIQualityResult(
        passed=len(errors) == 0,
        errors=tuple(errors),
        roi_pixel_count=roi_pixels,
        valid_pixel_count=valid_pixels,
        roi_fraction_of_valid_image=fraction,
    )


def select_review_samples(
    records: list[dict[str, Any]],
    samples_per_region: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Select reproducible samples from each anatomical region."""

    if samples_per_region < 0:
        raise ValueError("samples_per_region must be nonnegative.")

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer.")

    selected: list[dict[str, Any]] = []

    for region in ("forehead", "sternum"):
        candidates = [
            record
            for record in records
            if record.get("body_region") == region and record.get("passed") is True
        ]

        candidates = sorted(
            candidates,
            key=lambda item: str(item["relative_path"]),
        )

        # Independent deterministic RNG for each region.
        region_offset = 0 if region == "forehead" else 1

        rng = random.Random(random_seed + region_offset)

        count = min(
            samples_per_region,
            len(candidates),
        )

        sampled = rng.sample(
            candidates,
            count,
        )

        selected.extend(sampled)

    return selected


def summarize_roi_qc(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate image-level ROI QC outcomes."""

    total = len(records)

    passed = sum(record.get("passed") is True for record in records)

    failed = total - passed

    region_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}

    for record in records:
        region = str(record.get("body_region", "unknown"))

        region_counts[region] = region_counts.get(region, 0) + 1

        if record.get("passed") is not True:
            error_text = str(record.get("errors", "unknown_error"))

            for error in error_text.split(";"):
                error = error.strip() or "unknown_error"

                failure_counts[error] = failure_counts.get(error, 0) + 1

    return {
        "total_images": total,
        "passed_images": passed,
        "failed_images": failed,
        "pass_rate": (passed / total if total else 0.0),
        "region_counts": region_counts,
        "failure_counts": failure_counts,
        "status": ("passed" if total > 0 and failed == 0 else "failed"),
        "clinical_validation": False,
    }

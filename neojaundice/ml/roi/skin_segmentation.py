"""Experimental color-based skin segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

SUPPORTED_METHODS = ("ycrcb", "hsv")


@dataclass(frozen=True)
class ColorThreshold:
    """Lower and upper thresholds for a color space."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError("Color thresholds must contain three values.")

        for low, high in zip(
            self.lower,
            self.upper,
            strict=True,
        ):
            if (
                type(low) is not int
                or type(high) is not int
                or not 0 <= low <= high <= 255
            ):
                raise ValueError("Invalid color threshold range.")


@dataclass(frozen=True)
class SegmentationConfig:
    """Experimental segmentation configuration."""

    ycrcb: ColorThreshold
    hsv: ColorThreshold

    low_coverage_warning: float = 0.10
    high_coverage_warning: float = 0.95

    def __post_init__(self) -> None:
        if not (0 <= self.low_coverage_warning < self.high_coverage_warning <= 1):
            raise ValueError("Invalid segmentation coverage thresholds.")

        if self.hsv.lower[0] > 179:
            raise ValueError("OpenCV HSV hue must be between 0 and 179.")

        if self.hsv.upper[0] > 179:
            raise ValueError("OpenCV HSV hue must be between 0 and 179.")

    def threshold_for(
        self,
        method: str,
    ) -> ColorThreshold:
        if method == "ycrcb":
            return self.ycrcb

        if method == "hsv":
            return self.hsv

        raise ValueError(f"Unsupported segmentation method: {method}")


@dataclass(frozen=True)
class SegmentationResult:
    """Candidate skin mask and coverage statistics."""

    method: str
    mask: torch.Tensor
    selected_pixel_count: int
    geometric_roi_pixel_count: int
    coverage: float
    coverage_warning: str | None


def load_segmentation_config(
    path: Path,
) -> SegmentationConfig:
    """Load and validate segmentation thresholds."""

    if not path.is_file():
        raise FileNotFoundError(f"Segmentation config not found: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Segmentation config must be a mapping.")

    if raw.get("schema_version") != "1.0":
        raise ValueError("Unsupported segmentation schema version.")

    methods = raw.get("methods")

    if not isinstance(methods, dict):
        raise ValueError("Missing segmentation methods.")

    thresholds = {}

    for method in SUPPORTED_METHODS:
        settings = methods.get(method)

        if not isinstance(settings, dict):
            raise ValueError(f"Missing settings for {method}.")

        if settings.get("enabled") is not True:
            raise ValueError(f"Method {method} must be enabled.")

        lower = settings.get("lower")
        upper = settings.get("upper")

        if not (isinstance(lower, list) and isinstance(upper, list)):
            raise ValueError(f"Invalid threshold arrays for {method}.")

        thresholds[method] = ColorThreshold(
            lower=tuple(lower),
            upper=tuple(upper),
        )

    quality = raw.get("quality", {})

    return SegmentationConfig(
        ycrcb=thresholds["ycrcb"],
        hsv=thresholds["hsv"],
        low_coverage_warning=quality.get(
            "low_coverage_warning",
            0.10,
        ),
        high_coverage_warning=quality.get(
            "high_coverage_warning",
            0.95,
        ),
    )


class SkinSegmenter:
    """Generate candidate skin masks using color thresholds."""

    def __init__(
        self,
        config: SegmentationConfig,
    ) -> None:
        self.config = config

    def segment(
        self,
        preview_rgb: np.ndarray,
        geometric_roi_mask: torch.Tensor,
        method: str,
    ) -> SegmentationResult:
        """Apply color thresholding inside a geometric ROI."""

        threshold = self.config.threshold_for(method)

        if (
            preview_rgb.ndim != 3
            or preview_rgb.shape[2] != 3
            or preview_rgb.dtype != np.uint8
        ):
            raise ValueError("Expected uint8 RGB image.")

        height, width = preview_rgb.shape[:2]

        if tuple(geometric_roi_mask.shape) != (
            1,
            height,
            width,
        ):
            raise ValueError("Invalid geometric ROI mask shape.")

        if not torch.isfinite(geometric_roi_mask).all().item():
            raise ValueError("ROI mask contains non-finite values.")

        if (
            not torch.logical_or(
                geometric_roi_mask == 0,
                geometric_roi_mask == 1,
            )
            .all()
            .item()
        ):
            raise ValueError("Geometric ROI mask must be binary.")

        if method == "ycrcb":
            converted = cv2.cvtColor(
                preview_rgb,
                cv2.COLOR_RGB2YCrCb,
            )

        elif method == "hsv":
            converted = cv2.cvtColor(
                preview_rgb,
                cv2.COLOR_RGB2HSV,
            )

        else:
            raise ValueError(f"Unsupported method: {method}")

        lower = np.array(
            threshold.lower,
            dtype=np.uint8,
        )

        upper = np.array(
            threshold.upper,
            dtype=np.uint8,
        )

        color_mask = cv2.inRange(
            converted,
            lower,
            upper,
        )

        color_mask = (color_mask > 0).astype(np.float32)

        roi_numpy = geometric_roi_mask.detach().cpu().numpy()[0]

        final_mask = color_mask * roi_numpy

        selected_pixels = int(final_mask.sum())

        geometric_pixels = int(roi_numpy.sum())

        if geometric_pixels == 0:
            raise ValueError("Geometric ROI contains no pixels.")

        coverage = selected_pixels / geometric_pixels

        warning = None

        if coverage < self.config.low_coverage_warning:
            warning = "low_coverage"

        elif coverage > self.config.high_coverage_warning:
            warning = "high_coverage"

        return SegmentationResult(
            method=method,
            mask=torch.from_numpy(final_mask).unsqueeze(0),
            selected_pixel_count=selected_pixels,
            geometric_roi_pixel_count=geometric_pixels,
            coverage=coverage,
            coverage_warning=warning,
        )


def create_segmentation_overlay(
    preview_rgb: np.ndarray,
    segmentation_result: SegmentationResult,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay candidate segmentation pixels in green."""

    if (
        preview_rgb.ndim != 3
        or preview_rgb.shape[2] != 3
        or preview_rgb.dtype != np.uint8
    ):
        raise ValueError("Expected uint8 RGB preview.")

    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1.")

    mask = segmentation_result.mask.detach().cpu().numpy()[0].astype(bool)

    if mask.shape != preview_rgb.shape[:2]:
        raise ValueError("Segmentation mask shape mismatch.")

    overlay = preview_rgb.copy()

    green = np.array(
        [0, 255, 0],
        dtype=np.float32,
    )

    blended = (1 - alpha) * overlay[mask].astype(np.float32) + alpha * green

    overlay[mask] = blended.astype(np.uint8)

    return overlay

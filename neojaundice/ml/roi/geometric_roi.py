"""Geometric region-of-interest extraction for NeoJaundice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from ml.preprocessing.image_preprocessor import (
    PreprocessingResult,
)

SUPPORTED_REGIONS = ("forehead", "sternum")


@dataclass(frozen=True)
class ROIBox:
    """Normalized rectangle coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            raise ValueError("ROI coordinates must be finite numbers.")

        if not (0 <= self.x_min < self.x_max <= 1):
            raise ValueError("Invalid horizontal ROI coordinates.")

        if not (0 <= self.y_min < self.y_max <= 1):
            raise ValueError("Invalid vertical ROI coordinates.")


@dataclass(frozen=True)
class ROIConfig:
    """Region-specific geometric ROI configuration."""

    forehead: ROIBox
    sternum: ROIBox

    def get_box(self, region: str) -> ROIBox:
        if region == "forehead":
            return self.forehead

        if region == "sternum":
            return self.sternum

        raise ValueError(f"Unsupported anatomical region: {region}")


@dataclass(frozen=True)
class ROIResult:
    """Extracted ROI mask and geometry."""

    region: str
    roi_mask: torch.Tensor
    pixel_box: tuple[int, int, int, int]
    roi_pixel_count: int
    valid_pixel_count: int
    roi_fraction_of_valid_image: float
    roi_fraction_of_canvas: float


def load_roi_config(
    config_path: Path,
) -> ROIConfig:
    """Load and validate ROI configuration."""

    if not config_path.is_file():
        raise FileNotFoundError(f"ROI config not found: {config_path}")

    with config_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("ROI configuration must be a mapping.")

    if raw.get("schema_version") != "1.0":
        raise ValueError("Unsupported ROI schema version.")

    regions = raw.get("regions")

    if not isinstance(regions, dict):
        raise ValueError("Missing ROI region definitions.")

    boxes = {}

    for region in SUPPORTED_REGIONS:
        values = regions.get(region)

        if not isinstance(values, dict):
            raise ValueError(f"Missing ROI configuration for {region}.")

        required = {
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        }

        if not required.issubset(values):
            raise ValueError(f"Incomplete ROI coordinates for {region}.")

        boxes[region] = ROIBox(
            x_min=values["x_min"],
            y_min=values["y_min"],
            x_max=values["x_max"],
            y_max=values["y_max"],
        )

    return ROIConfig(
        forehead=boxes["forehead"],
        sternum=boxes["sternum"],
    )


class GeometricROIExtractor:
    """Extract region-specific rectangular candidate ROIs."""

    def __init__(
        self,
        config: ROIConfig,
    ) -> None:
        self.config = config

    def extract(
        self,
        preprocessing_result: PreprocessingResult,
        region: str,
    ) -> ROIResult:
        """Create a candidate ROI inside valid image content."""

        box = self.config.get_box(region)

        metadata = preprocessing_result.metadata

        width = metadata.resized_width
        height = metadata.resized_height

        if width <= 0 or height <= 0:
            raise ValueError("Invalid resized image dimensions.")

        left = metadata.padding_left
        top = metadata.padding_top

        x_min = left + int(math.floor(box.x_min * width))

        y_min = top + int(math.floor(box.y_min * height))

        x_max = left + int(math.ceil(box.x_max * width))

        y_max = top + int(math.ceil(box.y_max * height))

        x_min = max(
            left,
            min(x_min, left + width - 1),
        )

        y_min = max(
            top,
            min(y_min, top + height - 1),
        )

        x_max = max(
            x_min + 1,
            min(x_max, left + width),
        )

        y_max = max(
            y_min + 1,
            min(y_max, top + height),
        )

        valid_mask = preprocessing_result.valid_mask

        expected_shape = (
            1,
            metadata.target_height,
            metadata.target_width,
        )

        if tuple(valid_mask.shape) != expected_shape:
            raise ValueError("Invalid preprocessing valid-mask shape.")

        if not torch.isfinite(valid_mask).all().item():
            raise ValueError("Valid mask contains non-finite values.")

        if (
            not torch.logical_or(
                valid_mask == 0,
                valid_mask == 1,
            )
            .all()
            .item()
        ):
            raise ValueError("Valid mask must be binary.")

        rectangle_mask = torch.zeros_like(
            valid_mask,
            dtype=torch.float32,
        )

        rectangle_mask[
            :,
            y_min:y_max,
            x_min:x_max,
        ] = 1.0

        roi_mask = rectangle_mask * valid_mask

        roi_pixel_count = int(roi_mask.sum().item())

        valid_pixel_count = int(valid_mask.sum().item())

        if valid_pixel_count == 0:
            raise ValueError("No valid image pixels available.")

        if roi_pixel_count == 0:
            raise ValueError("ROI contains no valid image pixels.")

        canvas_pixels = metadata.target_width * metadata.target_height

        return ROIResult(
            region=region,
            roi_mask=roi_mask,
            pixel_box=(
                x_min,
                y_min,
                x_max,
                y_max,
            ),
            roi_pixel_count=roi_pixel_count,
            valid_pixel_count=valid_pixel_count,
            roi_fraction_of_valid_image=(roi_pixel_count / valid_pixel_count),
            roi_fraction_of_canvas=(roi_pixel_count / canvas_pixels),
        )


def create_roi_overlay(
    preview_rgb: np.ndarray,
    roi_result: ROIResult,
) -> np.ndarray:
    """Draw a red ROI rectangle on an RGB preview."""

    from PIL import Image, ImageDraw

    if (
        preview_rgb.ndim != 3
        or preview_rgb.shape[2] != 3
        or preview_rgb.dtype != np.uint8
    ):
        raise ValueError("Expected uint8 RGB preview.")

    image = Image.fromarray(preview_rgb.copy())

    draw = ImageDraw.Draw(image)

    x_min, y_min, x_max, y_max = roi_result.pixel_box

    draw.rectangle(
        (
            x_min,
            y_min,
            x_max - 1,
            y_max - 1,
        ),
        outline=(255, 0, 0),
        width=2,
    )

    return np.asarray(image).copy()

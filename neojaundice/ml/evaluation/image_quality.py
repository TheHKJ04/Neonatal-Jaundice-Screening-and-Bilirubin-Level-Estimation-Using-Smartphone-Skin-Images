"""Image quality metrics for neonatal skin image ROIs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ImageQualityConfig:
    underexposed_pixel_threshold: int = 30
    overexposed_pixel_threshold: int = 225

    minimum_roi_pixels: int = 100

    low_brightness_threshold: float = 40.0
    high_brightness_threshold: float = 215.0

    low_contrast_threshold: float = 15.0
    low_sharpness_threshold: float = 50.0

    maximum_underexposed_fraction: float = 0.20
    maximum_overexposed_fraction: float = 0.20


@dataclass(frozen=True)
class ImageQualityResult:
    roi_pixel_count: int

    mean_brightness: float
    brightness_std: float
    contrast: float
    laplacian_variance: float

    underexposed_fraction: float
    overexposed_fraction: float

    mean_red: float
    mean_green: float
    mean_blue: float

    std_red: float
    std_green: float
    std_blue: float

    warnings: tuple[str, ...]


def _validate_image(
    image_rgb: np.ndarray,
) -> None:
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a NumPy array.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    if image_rgb.dtype != np.uint8:
        raise ValueError("RGB image must use uint8 values.")


def _validate_mask(
    roi_mask: torch.Tensor,
    image_shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(roi_mask, torch.Tensor):
        raise TypeError("roi_mask must be a PyTorch tensor.")

    if roi_mask.ndim != 3 or roi_mask.shape[0] != 1:
        raise ValueError("ROI mask must have shape (1, H, W).")

    if not torch.isfinite(roi_mask).all().item():
        raise ValueError("ROI mask contains non-finite values.")

    if (
        not torch.logical_or(
            roi_mask == 0,
            roi_mask == 1,
        )
        .all()
        .item()
    ):
        raise ValueError("ROI mask must contain only binary values.")

    mask = roi_mask.detach().cpu().numpy()[0].astype(bool)

    if mask.shape != image_shape:
        raise ValueError("ROI mask dimensions do not match image.")

    if not np.any(mask):
        raise ValueError("ROI mask cannot be empty.")

    return mask


def calculate_image_quality(
    image_rgb: np.ndarray,
    roi_mask: torch.Tensor,
    config: ImageQualityConfig,
) -> ImageQualityResult:
    """Calculate exploratory image-quality metrics."""

    _validate_image(image_rgb)

    mask = _validate_mask(
        roi_mask,
        image_rgb.shape[:2],
    )

    roi_pixels = image_rgb[mask]

    roi_pixel_count = int(roi_pixels.shape[0])

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    gray_pixels = gray[mask].astype(np.float64)

    mean_brightness = float(np.mean(gray_pixels))

    brightness_std = float(np.std(gray_pixels))

    contrast = brightness_std

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    laplacian_variance = float(np.var(laplacian[mask]))

    underexposed_fraction = float(
        np.mean(gray_pixels <= config.underexposed_pixel_threshold)
    )

    overexposed_fraction = float(
        np.mean(gray_pixels >= config.overexposed_pixel_threshold)
    )

    rgb_mean = np.mean(
        roi_pixels,
        axis=0,
    )

    rgb_std = np.std(
        roi_pixels,
        axis=0,
    )

    warnings = []

    if roi_pixel_count < config.minimum_roi_pixels:
        warnings.append("insufficient_roi_pixels")

    if mean_brightness < config.low_brightness_threshold:
        warnings.append("low_brightness")

    if mean_brightness > config.high_brightness_threshold:
        warnings.append("high_brightness")

    if contrast < config.low_contrast_threshold:
        warnings.append("low_contrast")

    if laplacian_variance < config.low_sharpness_threshold:
        warnings.append("low_sharpness")

    if underexposed_fraction > config.maximum_underexposed_fraction:
        warnings.append("underexposed")

    if overexposed_fraction > config.maximum_overexposed_fraction:
        warnings.append("overexposed")

    return ImageQualityResult(
        roi_pixel_count=roi_pixel_count,
        mean_brightness=mean_brightness,
        brightness_std=brightness_std,
        contrast=contrast,
        laplacian_variance=laplacian_variance,
        underexposed_fraction=underexposed_fraction,
        overexposed_fraction=overexposed_fraction,
        mean_red=float(rgb_mean[0]),
        mean_green=float(rgb_mean[1]),
        mean_blue=float(rgb_mean[2]),
        std_red=float(rgb_std[0]),
        std_green=float(rgb_std[1]),
        std_blue=float(rgb_std[2]),
        warnings=tuple(warnings),
    )

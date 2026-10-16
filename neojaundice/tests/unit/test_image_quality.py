"""Unit tests for geometric ROI image-quality assessment."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.evaluation.image_quality import (
    ImageQualityConfig,
    calculate_image_quality,
)


@pytest.fixture
def config():
    return ImageQualityConfig()


@pytest.fixture
def full_mask():
    return torch.ones(
        (1, 10, 10),
        dtype=torch.float32,
    )


def make_image(value: int) -> np.ndarray:
    return np.full(
        (10, 10, 3),
        value,
        dtype=np.uint8,
    )


def test_valid_image(config, full_mask):
    image = make_image(120)

    result = calculate_image_quality(
        image,
        full_mask,
        config,
    )

    assert result.roi_pixel_count == 100
    assert result.mean_brightness == pytest.approx(120)


def test_empty_roi_rejected(config):
    image = make_image(120)

    mask = torch.zeros(
        (1, 10, 10),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        calculate_image_quality(
            image,
            mask,
            config,
        )


def test_invalid_image_dimensions(config, full_mask):
    image = np.zeros(
        (10, 10),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        calculate_image_quality(
            image,
            full_mask,
            config,
        )


def test_invalid_image_dtype(config, full_mask):
    image = np.zeros(
        (10, 10, 3),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        calculate_image_quality(
            image,
            full_mask,
            config,
        )


def test_invalid_mask_dimensions(config):
    image = make_image(120)

    mask = torch.ones(
        (10, 10),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        calculate_image_quality(
            image,
            mask,
            config,
        )


def test_nonbinary_mask(config, full_mask):
    image = make_image(120)

    mask = full_mask.clone()
    mask[0, 0, 0] = 0.5

    with pytest.raises(ValueError):
        calculate_image_quality(
            image,
            mask,
            config,
        )


def test_nonfinite_mask(config, full_mask):
    image = make_image(120)

    mask = full_mask.clone()
    mask[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        calculate_image_quality(
            image,
            mask,
            config,
        )


def test_low_brightness_warning(config, full_mask):
    image = make_image(20)

    result = calculate_image_quality(
        image,
        full_mask,
        config,
    )

    assert "low_brightness" in result.warnings


def test_high_brightness_warning(config, full_mask):
    image = make_image(240)

    result = calculate_image_quality(
        image,
        full_mask,
        config,
    )

    assert "high_brightness" in result.warnings


def test_underexposure_warning(config, full_mask):
    image = make_image(10)

    result = calculate_image_quality(
        image,
        full_mask,
        config,
    )

    assert "underexposed" in result.warnings


def test_overexposure_warning(config, full_mask):
    image = make_image(250)

    result = calculate_image_quality(
        image,
        full_mask,
        config,
    )

    assert "overexposed" in result.warnings


def test_low_contrast_warning(config, full_mask):
    image = make_image(120)

    result = calculate_image_quality(
        image,
        full_mask,
        config,
    )

    assert "low_contrast" in result.warnings


def test_insufficient_roi_pixels(config):
    image = make_image(120)

    mask = torch.zeros(
        (1, 10, 10),
        dtype=torch.float32,
    )

    mask[0, 0, 0] = 1

    result = calculate_image_quality(
        image,
        mask,
        config,
    )

    assert "insufficient_roi_pixels" in result.warnings


def test_rgb_channel_means(config, full_mask):
    image = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    image[:, :, 0] = 100
    image[:, :, 1] = 120
    image[:, :, 2] = 140

    result = calculate_image_quality(
        image,
        full_mask,
        config,
    )

    assert result.mean_red == pytest.approx(100)
    assert result.mean_green == pytest.approx(120)
    assert result.mean_blue == pytest.approx(140)


def test_result_values_are_finite(config, full_mask):
    image = make_image(120)

    result = calculate_image_quality(
        image,
        full_mask,
        config,
    )

    assert np.isfinite(result.laplacian_variance)

    assert np.isfinite(result.mean_brightness)

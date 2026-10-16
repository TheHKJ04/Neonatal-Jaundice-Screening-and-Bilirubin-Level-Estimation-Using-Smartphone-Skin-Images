"""Tests for ROI color feature extraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.features.color_features import extract_color_features


@pytest.fixture
def image():
    return np.full((10, 10, 3), 120, dtype=np.uint8)


@pytest.fixture
def mask():
    return torch.ones((1, 10, 10), dtype=torch.float32)


def test_feature_count(image, mask):
    result = extract_color_features(image, mask)
    assert len(result) == 61


def test_roi_pixel_count(image, mask):
    result = extract_color_features(image, mask)
    assert result["roi_pixel_count"] == 100


def test_rgb_mean(image, mask):
    result = extract_color_features(image, mask)
    assert result["rgb_r_mean"] == pytest.approx(120)


def test_rgb_std(image, mask):
    result = extract_color_features(image, mask)
    assert result["rgb_r_std"] == pytest.approx(0)


def test_rgb_median(image, mask):
    result = extract_color_features(image, mask)
    assert result["rgb_g_median"] == pytest.approx(120)


def test_percentiles(image, mask):
    result = extract_color_features(image, mask)
    assert result["rgb_b_p10"] == pytest.approx(120)
    assert result["rgb_b_p90"] == pytest.approx(120)


def test_all_features_finite(image, mask):
    result = extract_color_features(image, mask)
    assert all(np.isfinite(value) for value in result.values())


def test_empty_mask_rejected(image):
    mask = torch.zeros((1, 10, 10))
    with pytest.raises(ValueError):
        extract_color_features(image, mask)


def test_nonbinary_mask_rejected(image, mask):
    mask[0, 0, 0] = 0.5
    with pytest.raises(ValueError):
        extract_color_features(image, mask)


def test_nonfinite_mask_rejected(image, mask):
    mask[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        extract_color_features(image, mask)


def test_invalid_image_shape(mask):
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        extract_color_features(image, mask)


def test_invalid_image_dtype(mask):
    image = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        extract_color_features(image, mask)


def test_mask_shape_mismatch(image):
    mask = torch.ones((1, 5, 5))
    with pytest.raises(ValueError):
        extract_color_features(image, mask)


def test_roi_excludes_outside_pixels():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5, :, :] = 200

    mask = torch.zeros((1, 10, 10))
    mask[0, :5, :] = 1

    result = extract_color_features(image, mask)

    assert result["rgb_r_mean"] == pytest.approx(200)


def test_rgb_channel_values(mask):
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    image[:, :, 0] = 100
    image[:, :, 1] = 120
    image[:, :, 2] = 140

    result = extract_color_features(image, mask)

    assert result["rgb_r_mean"] == 100
    assert result["rgb_g_mean"] == 120
    assert result["rgb_b_mean"] == 140

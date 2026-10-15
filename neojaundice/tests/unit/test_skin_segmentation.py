"""Unit tests for experimental skin segmentation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ml.roi.segmentation_quality import (
    validate_segmentation_result,
)
from ml.roi.skin_segmentation import (
    ColorThreshold,
    SegmentationConfig,
    SkinSegmenter,
    create_segmentation_overlay,
    load_segmentation_config,
)


@pytest.fixture
def segmentation_config():
    return SegmentationConfig(
        ycrcb=ColorThreshold(
            lower=(0, 133, 77),
            upper=(255, 173, 127),
        ),
        hsv=ColorThreshold(
            lower=(0, 20, 40),
            upper=(25, 255, 255),
        ),
    )


@pytest.fixture
def sample_image():
    return np.full(
        (64, 64, 3),
        [200, 150, 120],
        dtype=np.uint8,
    )


@pytest.fixture
def roi_mask():
    mask = torch.zeros(
        (1, 64, 64),
        dtype=torch.float32,
    )

    mask[:, 10:50, 10:50] = 1.0

    return mask


def test_valid_threshold():
    threshold = ColorThreshold(
        (0, 10, 20),
        (255, 200, 220),
    )

    assert threshold.lower[0] == 0


def test_invalid_threshold_order():
    with pytest.raises(ValueError):
        ColorThreshold(
            (100, 10, 20),
            (50, 200, 220),
        )


def test_invalid_threshold_range():
    with pytest.raises(ValueError):
        ColorThreshold(
            (-1, 10, 20),
            (255, 200, 220),
        )


def test_invalid_hsv_hue():
    with pytest.raises(ValueError):
        SegmentationConfig(
            ycrcb=ColorThreshold(
                (0, 0, 0),
                (255, 255, 255),
            ),
            hsv=ColorThreshold(
                (0, 0, 0),
                (200, 255, 255),
            ),
        )


def test_unknown_method_rejected(segmentation_config):
    with pytest.raises(ValueError):
        segmentation_config.threshold_for("unknown")


@pytest.mark.parametrize(
    "method",
    ["ycrcb", "hsv"],
)
def test_segmentation_mask_shape(
    segmentation_config,
    sample_image,
    roi_mask,
    method,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        method,
    )

    assert result.mask.shape == (
        1,
        64,
        64,
    )


@pytest.mark.parametrize(
    "method",
    ["ycrcb", "hsv"],
)
def test_segmentation_mask_binary(
    segmentation_config,
    sample_image,
    roi_mask,
    method,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        method,
    )

    assert torch.logical_or(
        result.mask == 0,
        result.mask == 1,
    ).all()


@pytest.mark.parametrize(
    "method",
    ["ycrcb", "hsv"],
)
def test_segmentation_inside_roi(
    segmentation_config,
    sample_image,
    roi_mask,
    method,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        method,
    )

    assert torch.all(result.mask <= roi_mask)


@pytest.mark.parametrize(
    "method",
    ["ycrcb", "hsv"],
)
def test_segmentation_coverage_range(
    segmentation_config,
    sample_image,
    roi_mask,
    method,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        method,
    )

    assert 0 <= result.coverage <= 1


def test_invalid_image_dtype(
    segmentation_config,
    roi_mask,
):
    image = np.ones(
        (64, 64, 3),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        SkinSegmenter(segmentation_config).segment(
            image,
            roi_mask,
            "hsv",
        )


def test_invalid_roi_shape(
    segmentation_config,
    sample_image,
):
    mask = torch.zeros(
        (1, 32, 32),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        SkinSegmenter(segmentation_config).segment(
            sample_image,
            mask,
            "hsv",
        )


def test_zero_roi_rejected(
    segmentation_config,
    sample_image,
):
    mask = torch.zeros(
        (1, 64, 64),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        SkinSegmenter(segmentation_config).segment(
            sample_image,
            mask,
            "hsv",
        )


def test_nonbinary_roi_rejected(
    segmentation_config,
    sample_image,
    roi_mask,
):
    mask = roi_mask.clone()
    mask[0, 20, 20] = 0.5

    with pytest.raises(ValueError):
        SkinSegmenter(segmentation_config).segment(
            sample_image,
            mask,
            "hsv",
        )


def test_segmentation_qc_passes(
    segmentation_config,
    sample_image,
    roi_mask,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        "hsv",
    )

    qc = validate_segmentation_result(
        result,
        roi_mask,
    )

    assert qc.passed is True


def test_segmentation_qc_detects_count_mismatch(
    segmentation_config,
    sample_image,
    roi_mask,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        "hsv",
    )

    modified = replace(
        result,
        selected_pixel_count=999999,
    )

    qc = validate_segmentation_result(
        modified,
        roi_mask,
    )

    assert "selected_pixel_count_mismatch" in qc.errors


def test_segmentation_qc_detects_coverage_mismatch(
    segmentation_config,
    sample_image,
    roi_mask,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        "hsv",
    )

    modified = replace(
        result,
        coverage=0.123456,
    )

    qc = validate_segmentation_result(
        modified,
        roi_mask,
    )

    assert "coverage_mismatch" in qc.errors


def test_segmentation_qc_detects_outside_roi(
    segmentation_config,
    sample_image,
    roi_mask,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        "hsv",
    )

    mask = result.mask.clone()
    mask[0, 0, 0] = 1.0

    modified = replace(
        result,
        mask=mask,
    )

    qc = validate_segmentation_result(
        modified,
        roi_mask,
    )

    assert "mask_outside_geometric_roi" in qc.errors


def test_overlay_shape(
    segmentation_config,
    sample_image,
    roi_mask,
):
    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        "hsv",
    )

    overlay = create_segmentation_overlay(
        sample_image,
        result,
    )

    assert overlay.shape == (
        64,
        64,
        3,
    )


def test_overlay_preserves_input(
    segmentation_config,
    sample_image,
    roi_mask,
):
    original = sample_image.copy()

    result = SkinSegmenter(segmentation_config).segment(
        sample_image,
        roi_mask,
        "hsv",
    )

    create_segmentation_overlay(
        sample_image,
        result,
    )

    assert np.array_equal(
        sample_image,
        original,
    )


def test_load_config(tmp_path: Path):
    config_path = tmp_path / "segmentation.yaml"

    raw = {
        "schema_version": "1.0",
        "methods": {
            "ycrcb": {
                "enabled": True,
                "lower": [0, 133, 77],
                "upper": [255, 173, 127],
            },
            "hsv": {
                "enabled": True,
                "lower": [0, 20, 40],
                "upper": [25, 255, 255],
            },
        },
    }

    config_path.write_text(
        yaml.safe_dump(raw),
        encoding="utf-8",
    )

    config = load_segmentation_config(config_path)

    assert config.hsv.upper[0] == 25


def test_missing_config(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_segmentation_config(tmp_path / "missing.yaml")

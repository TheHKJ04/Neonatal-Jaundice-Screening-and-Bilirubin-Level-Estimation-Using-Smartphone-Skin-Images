"""Unit tests for geometric ROI extraction."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
)
from ml.roi.geometric_roi import (
    GeometricROIExtractor,
    ROIBox,
    ROIConfig,
    create_roi_overlay,
    load_roi_config,
)


@pytest.fixture
def preprocessing_result(tmp_path: Path):
    path = tmp_path / "sample.png"

    Image.new(
        "RGB",
        (640, 480),
        color=(200, 150, 100),
    ).save(path)

    return ImagePreprocessor(PreprocessingConfig(normalization="unit")).process(path)


@pytest.fixture
def roi_config():
    return ROIConfig(
        forehead=ROIBox(
            x_min=0.30,
            y_min=0.25,
            x_max=0.70,
            y_max=0.60,
        ),
        sternum=ROIBox(
            x_min=0.30,
            y_min=0.30,
            x_max=0.70,
            y_max=0.70,
        ),
    )


def test_valid_roi_box():
    box = ROIBox(
        0.1,
        0.2,
        0.8,
        0.9,
    )

    assert box.x_min == 0.1


def test_invalid_horizontal_coordinates():
    with pytest.raises(ValueError):
        ROIBox(
            0.8,
            0.2,
            0.3,
            0.9,
        )


def test_invalid_vertical_coordinates():
    with pytest.raises(ValueError):
        ROIBox(
            0.1,
            0.9,
            0.8,
            0.2,
        )


def test_out_of_range_coordinates():
    with pytest.raises(ValueError):
        ROIBox(
            -0.1,
            0.2,
            0.8,
            0.9,
        )


def test_nan_coordinates():
    with pytest.raises(ValueError):
        ROIBox(
            float("nan"),
            0.2,
            0.8,
            0.9,
        )


def test_unknown_region_rejected(roi_config):
    with pytest.raises(ValueError):
        roi_config.get_box("unknown")


def test_forehead_roi_mask_shape(
    preprocessing_result,
    roi_config,
):
    result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        region="forehead",
    )

    assert result.roi_mask.shape == (
        1,
        224,
        224,
    )


def test_sternum_roi_mask_shape(
    preprocessing_result,
    roi_config,
):
    result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        region="sternum",
    )

    assert result.roi_mask.shape == (
        1,
        224,
        224,
    )


def test_roi_mask_is_binary(
    preprocessing_result,
    roi_config,
):
    result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        region="forehead",
    )

    assert torch.logical_or(
        result.roi_mask == 0,
        result.roi_mask == 1,
    ).all()


def test_roi_excludes_padding(
    preprocessing_result,
    roi_config,
):
    result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        region="forehead",
    )

    assert result.roi_mask[:, :28, :].sum() == 0
    assert result.roi_mask[:, 196:, :].sum() == 0


def test_roi_has_positive_pixel_count(
    preprocessing_result,
    roi_config,
):
    result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        region="forehead",
    )

    assert result.roi_pixel_count > 0


def test_roi_coverage_between_zero_and_one(
    preprocessing_result,
    roi_config,
):
    result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        region="forehead",
    )

    assert 0 < result.roi_fraction_of_valid_image <= 1


def test_roi_inside_valid_mask(
    preprocessing_result,
    roi_config,
):
    result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        region="forehead",
    )

    assert torch.all(result.roi_mask <= preprocessing_result.valid_mask)


def test_forehead_pixel_coordinates(
    preprocessing_result,
    roi_config,
):
    result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        region="forehead",
    )

    assert result.pixel_box == (
        67,
        70,
        157,
        129,
    )


def test_different_region_coordinates(
    preprocessing_result,
    roi_config,
):
    extractor = GeometricROIExtractor(roi_config)

    forehead = extractor.extract(
        preprocessing_result,
        "forehead",
    )

    sternum = extractor.extract(
        preprocessing_result,
        "sternum",
    )

    assert forehead.pixel_box != sternum.pixel_box


def test_invalid_valid_mask_shape(
    preprocessing_result,
    roi_config,
):
    preprocessing_result.valid_mask = preprocessing_result.valid_mask[:, :100, :]

    with pytest.raises(ValueError):
        GeometricROIExtractor(roi_config).extract(
            preprocessing_result,
            "forehead",
        )


def test_zero_valid_mask_rejected(
    preprocessing_result,
    roi_config,
):
    preprocessing_result.valid_mask.zero_()

    with pytest.raises(ValueError):
        GeometricROIExtractor(roi_config).extract(
            preprocessing_result,
            "forehead",
        )


def test_overlay_shape(
    preprocessing_result,
    roi_config,
):
    roi_result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        "forehead",
    )

    overlay = create_roi_overlay(
        preprocessing_result.preview_rgb,
        roi_result,
    )

    assert overlay.shape == (
        224,
        224,
        3,
    )

    assert overlay.dtype == np.uint8


def test_overlay_draws_red_rectangle(
    preprocessing_result,
    roi_config,
):
    roi_result = GeometricROIExtractor(roi_config).extract(
        preprocessing_result,
        "forehead",
    )

    overlay = create_roi_overlay(
        preprocessing_result.preview_rgb,
        roi_result,
    )

    x_min, y_min, _, _ = roi_result.pixel_box

    assert tuple(overlay[y_min, x_min]) == (255, 0, 0)


def test_load_roi_yaml(tmp_path: Path):
    path = tmp_path / "roi.yaml"

    data = {
        "schema_version": "1.0",
        "regions": {
            "forehead": {
                "x_min": 0.2,
                "y_min": 0.2,
                "x_max": 0.8,
                "y_max": 0.8,
            },
            "sternum": {
                "x_min": 0.3,
                "y_min": 0.3,
                "x_max": 0.7,
                "y_max": 0.7,
            },
        },
    }

    path.write_text(
        yaml.safe_dump(data),
        encoding="utf-8",
    )

    config = load_roi_config(path)

    assert config.forehead.x_min == 0.2
    assert config.sternum.x_max == 0.7


def test_missing_roi_yaml(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_roi_config(tmp_path / "missing.yaml")

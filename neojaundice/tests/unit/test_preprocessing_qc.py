"""Tests for preprocessing quality-control validation."""

from pathlib import Path

import pytest
from PIL import Image

from ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
)
from ml.preprocessing.quality_control import (
    summarize_qc_results,
    validate_preprocessing_result,
)


@pytest.fixture
def valid_result(tmp_path: Path):
    path = tmp_path / "sample.png"

    Image.new(
        "RGB",
        (640, 480),
        color=(200, 150, 100),
    ).save(path)

    config = PreprocessingConfig(normalization="unit")

    result = ImagePreprocessor(config).process(path)

    return result, config


def test_valid_result_passes(valid_result):
    result, config = valid_result

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert qc.passed is True
    assert qc.errors == ()


def test_valid_pixel_fraction(valid_result):
    result, config = valid_result

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert qc.valid_pixel_fraction == pytest.approx(0.75)


def test_invalid_image_shape_detected(valid_result):
    result, config = valid_result

    result.image_tensor = result.image_tensor[:2]

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "invalid_image_shape" in qc.errors


def test_invalid_mask_shape_detected(valid_result):
    result, config = valid_result

    result.valid_mask = result.valid_mask[:, :100, :]

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "invalid_mask_shape" in qc.errors


def test_non_finite_image_detected(valid_result):
    result, config = valid_result

    result.image_tensor[0, 0, 0] = float("nan")

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "non_finite_image_values" in qc.errors


def test_non_binary_mask_detected(valid_result):
    result, config = valid_result

    result.valid_mask[0, 30, 30] = 0.5

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "non_binary_mask" in qc.errors


def test_mask_geometry_mismatch_detected(valid_result):
    result, config = valid_result

    result.valid_mask[0, 30, 30] = 0.0

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "mask_geometry_mismatch" in qc.errors


def test_tensor_preview_mismatch_detected(valid_result):
    result, config = valid_result

    result.image_tensor[0, 40, 40] = 0.0

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "tensor_preview_mismatch" in qc.errors


def test_summary_all_passed():
    records = [
        {
            "body_region": "forehead",
            "passed": True,
        },
        {
            "body_region": "sternum",
            "passed": True,
        },
    ]

    summary = summarize_qc_results(records)

    assert summary["total_images"] == 2
    assert summary["passed_images"] == 2
    assert summary["failed_images"] == 0
    assert summary["status"] == "passed"


def test_summary_partial_failure():
    records = [
        {
            "body_region": "forehead",
            "passed": True,
        },
        {
            "body_region": "sternum",
            "passed": False,
        },
    ]

    summary = summarize_qc_results(records)

    assert summary["failed_images"] == 1
    assert summary["status"] == "failed"


def test_summary_empty_records():
    summary = summarize_qc_results([])

    assert summary["total_images"] == 0
    assert summary["status"] == "failed"


def test_region_counts():
    records = [
        {
            "body_region": "forehead",
            "passed": True,
        },
        {
            "body_region": "forehead",
            "passed": True,
        },
        {
            "body_region": "sternum",
            "passed": False,
        },
    ]

    summary = summarize_qc_results(records)

    assert summary["region_counts"] == {
        "forehead": 2,
        "sternum": 1,
    }


def test_invalid_image_dtype_detected(valid_result):
    result, config = valid_result

    result.image_tensor = result.image_tensor.double()

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "invalid_image_dtype" in qc.errors


def test_invalid_mask_dtype_detected(valid_result):
    result, config = valid_result

    result.valid_mask = result.valid_mask.double()

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "invalid_mask_dtype" in qc.errors


def test_non_finite_mask_detected(valid_result):
    result, config = valid_result

    result.valid_mask[0, 30, 30] = float("inf")

    qc = validate_preprocessing_result(
        result,
        config,
    )

    assert "non_finite_mask_values" in qc.errors

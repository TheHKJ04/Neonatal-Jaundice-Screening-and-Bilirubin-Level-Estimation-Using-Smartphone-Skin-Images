"""Unit tests for ROI quality control."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
)
from ml.roi.geometric_roi import (
    GeometricROIExtractor,
    ROIBox,
    ROIConfig,
)
from ml.roi.quality_control import (
    select_review_samples,
    summarize_roi_qc,
    validate_roi_result,
)


@pytest.fixture
def valid_pair(tmp_path: Path):
    image_path = tmp_path / "sample.png"

    Image.new(
        "RGB",
        (640, 480),
        color=(200, 150, 100),
    ).save(image_path)

    preprocessing = ImagePreprocessor(
        PreprocessingConfig(normalization="unit")
    ).process(image_path)

    roi_config = ROIConfig(
        forehead=ROIBox(
            0.30,
            0.25,
            0.70,
            0.60,
        ),
        sternum=ROIBox(
            0.30,
            0.30,
            0.70,
            0.70,
        ),
    )

    roi = GeometricROIExtractor(roi_config).extract(
        preprocessing,
        "forehead",
    )

    return preprocessing, roi


def test_valid_roi_passes(valid_pair):
    preprocessing, roi = valid_pair

    qc = validate_roi_result(
        preprocessing,
        roi,
    )

    assert qc.passed is True
    assert qc.errors == ()


def test_roi_pixel_count_positive(valid_pair):
    preprocessing, roi = valid_pair

    qc = validate_roi_result(
        preprocessing,
        roi,
    )

    assert qc.roi_pixel_count > 0


def test_valid_pixel_count_positive(valid_pair):
    preprocessing, roi = valid_pair

    qc = validate_roi_result(
        preprocessing,
        roi,
    )

    assert qc.valid_pixel_count > 0


def test_roi_coverage_consistent(valid_pair):
    preprocessing, roi = valid_pair

    qc = validate_roi_result(
        preprocessing,
        roi,
    )

    assert qc.roi_fraction_of_valid_image == (
        pytest.approx(roi.roi_fraction_of_valid_image)
    )


def test_empty_roi_detected(valid_pair):
    preprocessing, roi = valid_pair

    modified = replace(
        roi,
        roi_mask=torch.zeros_like(roi.roi_mask),
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "empty_roi" in qc.errors


def test_roi_padding_contamination_detected(valid_pair):
    preprocessing, roi = valid_pair

    mask = roi.roi_mask.clone()
    mask[0, 0, 0] = 1.0

    modified = replace(
        roi,
        roi_mask=mask,
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "roi_includes_padding" in qc.errors


def test_non_binary_roi_detected(valid_pair):
    preprocessing, roi = valid_pair

    mask = roi.roi_mask.clone()
    mask[0, 80, 80] = 0.5

    modified = replace(
        roi,
        roi_mask=mask,
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "non_binary_roi_mask" in qc.errors


def test_non_finite_roi_detected(valid_pair):
    preprocessing, roi = valid_pair

    mask = roi.roi_mask.clone()
    mask[0, 80, 80] = float("nan")

    modified = replace(
        roi,
        roi_mask=mask,
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "non_finite_roi_mask" in qc.errors


def test_incorrect_pixel_count_detected(valid_pair):
    preprocessing, roi = valid_pair

    modified = replace(
        roi,
        roi_pixel_count=1,
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "roi_pixel_count_mismatch" in qc.errors


def test_incorrect_valid_count_detected(valid_pair):
    preprocessing, roi = valid_pair

    modified = replace(
        roi,
        valid_pixel_count=1,
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "valid_pixel_count_mismatch" in qc.errors


def test_invalid_roi_box_detected(valid_pair):
    preprocessing, roi = valid_pair

    modified = replace(
        roi,
        pixel_box=(-1, 10, 100, 100),
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "invalid_roi_box" in qc.errors


def test_roi_geometry_mismatch_detected(valid_pair):
    preprocessing, roi = valid_pair

    mask = roi.roi_mask.clone()
    mask[0, 80, 80] = 0.0

    modified = replace(
        roi,
        roi_mask=mask,
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "roi_geometry_mismatch" in qc.errors


def test_roi_coverage_mismatch_detected(valid_pair):
    preprocessing, roi = valid_pair

    modified = replace(
        roi,
        roi_fraction_of_valid_image=0.99,
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "roi_coverage_mismatch" in qc.errors


def test_canvas_coverage_mismatch_detected(valid_pair):
    preprocessing, roi = valid_pair

    modified = replace(
        roi,
        roi_fraction_of_canvas=0.99,
    )

    qc = validate_roi_result(
        preprocessing,
        modified,
    )

    assert "canvas_coverage_mismatch" in qc.errors


@pytest.fixture
def sample_records():
    records = []

    for region in ("forehead", "sternum"):
        for index in range(30):
            records.append(
                {
                    "relative_path": (f"{region}/image_{index:03d}.jpg"),
                    "body_region": region,
                    "passed": True,
                }
            )

    return records


def test_sampling_is_deterministic(sample_records):
    first = select_review_samples(
        sample_records,
        samples_per_region=10,
        random_seed=42,
    )

    second = select_review_samples(
        sample_records,
        samples_per_region=10,
        random_seed=42,
    )

    assert first == second


def test_sampling_is_region_balanced(sample_records):
    selected = select_review_samples(
        sample_records,
        samples_per_region=10,
        random_seed=42,
    )

    forehead = sum(item["body_region"] == "forehead" for item in selected)

    sternum = sum(item["body_region"] == "sternum" for item in selected)

    assert forehead == 10
    assert sternum == 10


def test_sampling_excludes_failures(sample_records):
    sample_records[0]["passed"] = False

    selected = select_review_samples(
        sample_records,
        samples_per_region=10,
        random_seed=42,
    )

    assert all(item["passed"] is True for item in selected)


def test_sampling_does_not_exceed_available(sample_records):
    selected = select_review_samples(
        sample_records,
        samples_per_region=100,
        random_seed=42,
    )

    assert len(selected) == 60


def test_negative_sample_count_rejected(sample_records):
    with pytest.raises(ValueError):
        select_review_samples(
            sample_records,
            samples_per_region=-1,
            random_seed=42,
        )


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

    summary = summarize_roi_qc(records)

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
            "errors": "invalid_roi_box",
        },
    ]

    summary = summarize_roi_qc(records)

    assert summary["failed_images"] == 1
    assert summary["status"] == "failed"


def test_summary_empty():
    summary = summarize_roi_qc([])

    assert summary["total_images"] == 0
    assert summary["status"] == "failed"


def test_failure_counts():
    records = [
        {
            "body_region": "forehead",
            "passed": False,
            "errors": ("invalid_roi_box;empty_roi"),
        }
    ]

    summary = summarize_roi_qc(records)

    assert summary["failure_counts"] == {
        "invalid_roi_box": 1,
        "empty_roi": 1,
    }

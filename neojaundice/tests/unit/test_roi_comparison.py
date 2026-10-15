"""Unit tests for ROI method comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.roi_comparison import (
    ComparisonConfig,
    bootstrap_mean_interval,
    calculate_coverage_statistics,
    calculate_paired_comparison,
    create_comparison_summary,
    validate_manifest,
)


@pytest.fixture
def sample_manifest():
    return pd.DataFrame(
        [
            {
                "relative_path": "forehead/a.jpg",
                "body_region": "forehead",
                "method": "ycrcb",
                "passed": True,
                "coverage": 0.80,
            },
            {
                "relative_path": "forehead/a.jpg",
                "body_region": "forehead",
                "method": "hsv",
                "passed": True,
                "coverage": 0.90,
            },
            {
                "relative_path": "sternum/b.jpg",
                "body_region": "sternum",
                "method": "ycrcb",
                "passed": True,
                "coverage": 0.70,
            },
            {
                "relative_path": "sternum/b.jpg",
                "body_region": "sternum",
                "method": "hsv",
                "passed": True,
                "coverage": 0.75,
            },
        ]
    )


@pytest.fixture
def comparison_config():
    return ComparisonConfig(
        bootstrap_iterations=100,
        confidence_level=0.95,
        random_seed=42,
        disagreement_threshold=0.05,
    )


def test_valid_manifest(sample_manifest):
    result = validate_manifest(sample_manifest)

    assert len(result) == 4


def test_missing_column_rejected(sample_manifest):
    modified = sample_manifest.drop(columns=["coverage"])

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_empty_manifest_rejected():
    empty = pd.DataFrame(
        columns=[
            "relative_path",
            "body_region",
            "method",
            "passed",
            "coverage",
        ]
    )

    with pytest.raises(ValueError):
        validate_manifest(empty)


def test_duplicate_evaluation_rejected(sample_manifest):
    modified = pd.concat(
        [
            sample_manifest,
            sample_manifest.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_missing_method_rejected(sample_manifest):
    modified = sample_manifest.iloc[:-1]

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_invalid_coverage_rejected(sample_manifest):
    modified = sample_manifest.copy()
    modified.loc[0, "coverage"] = 1.5

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_negative_coverage_rejected(sample_manifest):
    modified = sample_manifest.copy()
    modified.loc[0, "coverage"] = -0.1

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_nonfinite_coverage_rejected(sample_manifest):
    modified = sample_manifest.copy()
    modified.loc[0, "coverage"] = np.nan

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_failed_evaluation_rejected(sample_manifest):
    modified = sample_manifest.copy()
    modified.loc[0, "passed"] = False

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_unknown_method_rejected(sample_manifest):
    modified = sample_manifest.copy()
    modified.loc[0, "method"] = "unknown"

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_conflicting_regions_rejected(sample_manifest):
    modified = sample_manifest.copy()
    modified.loc[1, "body_region"] = "sternum"

    with pytest.raises(ValueError):
        validate_manifest(modified)


def test_bootstrap_is_reproducible(comparison_config):
    values = np.array([0.7, 0.8, 0.9, 1.0])

    first = bootstrap_mean_interval(
        values,
        comparison_config,
    )

    second = bootstrap_mean_interval(
        values,
        comparison_config,
    )

    assert first == second


def test_bootstrap_rejects_empty(comparison_config):
    with pytest.raises(ValueError):
        bootstrap_mean_interval(
            np.array([]),
            comparison_config,
        )


def test_bootstrap_rejects_nan(comparison_config):
    with pytest.raises(ValueError):
        bootstrap_mean_interval(
            np.array([0.8, np.nan]),
            comparison_config,
        )


def test_invalid_bootstrap_iterations():
    with pytest.raises(ValueError):
        ComparisonConfig(bootstrap_iterations=10)


def test_invalid_confidence_level():
    with pytest.raises(ValueError):
        ComparisonConfig(confidence_level=1.0)


def test_coverage_statistics(
    sample_manifest,
    comparison_config,
):
    validated = validate_manifest(sample_manifest)

    statistics = calculate_coverage_statistics(
        validated,
        comparison_config,
    )

    assert len(statistics) == 4


def test_paired_comparison_count(
    sample_manifest,
    comparison_config,
):
    validated = validate_manifest(sample_manifest)

    paired = calculate_paired_comparison(
        validated,
        comparison_config,
    )

    assert len(paired) == 2


def test_paired_difference(
    sample_manifest,
    comparison_config,
):
    validated = validate_manifest(sample_manifest)

    paired = calculate_paired_comparison(
        validated,
        comparison_config,
    )

    forehead = paired[paired["body_region"] == "forehead"].iloc[0]

    assert forehead["coverage_difference_hsv_minus_ycrcb"] == pytest.approx(0.10)


def test_paired_summary(
    sample_manifest,
    comparison_config,
):
    validated = validate_manifest(sample_manifest)

    paired = calculate_paired_comparison(
        validated,
        comparison_config,
    )

    summary = create_comparison_summary(
        paired,
        comparison_config,
    )

    assert summary["total_images"] == 2


def test_summary_has_research_limitations(
    sample_manifest,
    comparison_config,
):
    validated = validate_manifest(sample_manifest)

    paired = calculate_paired_comparison(
        validated,
        comparison_config,
    )

    summary = create_comparison_summary(
        paired,
        comparison_config,
    )

    assert summary["clinical_validation"] is False

    assert summary["patient_independence_verified"] is False


def test_summary_rejects_empty(comparison_config):
    with pytest.raises(ValueError):
        create_comparison_summary(
            pd.DataFrame(),
            comparison_config,
        )

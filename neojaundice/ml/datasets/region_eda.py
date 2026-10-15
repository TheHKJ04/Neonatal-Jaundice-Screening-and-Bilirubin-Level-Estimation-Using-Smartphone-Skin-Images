"""Region-specific exploratory analysis for neonatal skin images.

This module consumes previously generated image-level statistics.
It does not decode images or infer clinical bilirubin measurements.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

REGIONS = ("forehead", "sternum")

METRICS = (
    "brightness",
    "contrast",
    "blur_score",
    "red_mean",
    "green_mean",
    "blue_mean",
    "saturation_mean",
    "overexposed_fraction",
    "underexposed_fraction",
)

REQUIRED_COLUMNS = {
    "relative_path",
    "body_region_folder",
    *METRICS,
}


def load_image_statistics(csv_path: Path) -> pd.DataFrame:
    """Load and validate an image statistics manifest."""

    if not csv_path.is_file():
        raise FileNotFoundError(f"Image statistics CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    if dataframe.empty:
        raise ValueError("Image statistics CSV is empty.")

    if dataframe["relative_path"].isna().any():
        raise ValueError("Image paths must not be missing.")

    if dataframe["relative_path"].duplicated().any():
        raise ValueError("Duplicate relative image paths were detected.")

    if dataframe["body_region_folder"].isna().any():
        raise ValueError("Body-region values must not be missing.")

    unknown_regions = set(dataframe["body_region_folder"].unique()) - set(REGIONS)

    if unknown_regions:
        raise ValueError(
            "Unsupported body regions: "
            + ", ".join(sorted(str(x) for x in unknown_regions))
        )

    for metric in METRICS:
        converted = pd.to_numeric(
            dataframe[metric],
            errors="coerce",
        )

        if converted.isna().any():
            raise ValueError(f"Metric '{metric}' contains missing or invalid values.")

        if not converted.map(math.isfinite).all():
            raise ValueError(f"Metric '{metric}' contains non-finite values.")

        dataframe[metric] = converted.astype(float)

    for metric in (
        "overexposed_fraction",
        "underexposed_fraction",
    ):
        if not dataframe[metric].between(0.0, 1.0).all():
            raise ValueError(f"Metric '{metric}' must be between 0 and 1.")

    return dataframe


def calculate_metric_statistics(
    values: pd.Series,
) -> dict[str, float]:
    """Calculate descriptive statistics for one metric."""

    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
    }


def calculate_region_statistics(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate descriptive statistics for each anatomical region."""

    results: dict[str, Any] = {}

    for region in REGIONS:
        subset = dataframe.loc[dataframe["body_region_folder"] == region]

        if subset.empty:
            raise ValueError(f"No images found for region: {region}")

        results[region] = {
            "image_count": int(len(subset)),
            "metrics": {
                metric: calculate_metric_statistics(subset[metric])
                for metric in METRICS
            },
        }

    return results


def build_region_comparison(
    region_statistics: dict[str, Any],
) -> pd.DataFrame:
    """Build a table of descriptive differences between regions.

    Differences are forehead minus sternum.

    These are not clinical effect estimates or significance tests.
    """

    rows: list[dict[str, Any]] = []

    forehead = region_statistics["forehead"]["metrics"]
    sternum = region_statistics["sternum"]["metrics"]

    for metric in METRICS:
        forehead_mean = forehead[metric]["mean"]
        sternum_mean = sternum[metric]["mean"]

        rows.append(
            {
                "metric": metric,
                "forehead_mean": forehead_mean,
                "sternum_mean": sternum_mean,
                "mean_difference": (forehead_mean - sternum_mean),
                "forehead_median": forehead[metric]["median"],
                "sternum_median": sternum[metric]["median"],
                "forehead_std": forehead[metric]["std"],
                "sternum_std": sternum[metric]["std"],
            }
        )

    return pd.DataFrame(rows)


def build_region_report(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Create JSON-compatible statistics and a comparison table."""

    region_statistics = calculate_region_statistics(dataframe)

    comparison = build_region_comparison(region_statistics)

    report = {
        "analysis_type": "region_specific_image_eda",
        "total_images": int(len(dataframe)),
        "regions": region_statistics,
        "comparison_definition": ("mean_difference = forehead_mean - sternum_mean"),
        "clinical_labels_available": False,
        "patient_pairing_verified": False,
        "limitations": [
            "The analysis is descriptive only.",
            "No numerical bilirubin references are available.",
            "Patient-level image pairing is unverified.",
            "No causal interpretation is supported.",
            "Observed differences may reflect acquisition conditions.",
        ],
    }

    return report, comparison


def save_region_reports(
    report: dict[str, Any],
    comparison: pd.DataFrame,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Save CSV and JSON reports."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_directory / "region_comparison.csv"

    json_path = output_directory / "region_comparison.json"

    comparison.to_csv(
        csv_path,
        index=False,
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return csv_path, json_path

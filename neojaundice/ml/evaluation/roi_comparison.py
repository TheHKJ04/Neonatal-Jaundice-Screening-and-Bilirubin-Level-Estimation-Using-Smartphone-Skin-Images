"""Statistical comparison of experimental ROI segmentation methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "relative_path",
    "body_region",
    "method",
    "passed",
    "coverage",
}

EXPECTED_METHODS = ("ycrcb", "hsv")
EXPECTED_REGIONS = ("forehead", "sternum")


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for ROI coverage analysis."""

    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95
    random_seed: int = 42
    near_full_coverage_threshold: float = 0.95
    disagreement_threshold: float = 0.05

    def __post_init__(self) -> None:
        if (
            type(self.bootstrap_iterations) is not int
            or self.bootstrap_iterations < 100
        ):
            raise ValueError("bootstrap_iterations must be at least 100.")

        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")

        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer.")

        for name in (
            "near_full_coverage_threshold",
            "disagreement_threshold",
        ):
            value = getattr(self, name)

            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")


def validate_manifest(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize a segmentation manifest."""

    missing = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    if dataframe.empty:
        raise ValueError("Segmentation manifest is empty.")

    validated = dataframe.copy()

    validated["relative_path"] = validated["relative_path"].astype(str)

    validated["body_region"] = validated["body_region"].astype(str)

    validated["method"] = validated["method"].astype(str)

    if validated["relative_path"].str.strip().eq("").any():
        raise ValueError("Empty image path found.")

    if not validated["method"].isin(EXPECTED_METHODS).all():
        raise ValueError("Unexpected segmentation method.")

    if not validated["body_region"].isin(EXPECTED_REGIONS).all():
        raise ValueError("Unexpected anatomical region.")

    passed_values = validated["passed"].astype(str).str.strip().str.lower()

    if not passed_values.isin(("true", "false")).all():
        raise ValueError("Invalid passed field.")

    validated["passed"] = passed_values.eq("true")

    validated["coverage"] = pd.to_numeric(
        validated["coverage"],
        errors="coerce",
    )

    if not np.isfinite(validated["coverage"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite coverage found.")

    if not validated["coverage"].between(0, 1).all():
        raise ValueError("Coverage must be between 0 and 1.")

    duplicates = validated.duplicated(
        subset=["relative_path", "method"],
    )

    if duplicates.any():
        raise ValueError("Duplicate image-method evaluations found.")

    region_counts = validated.groupby("relative_path")["body_region"].nunique()

    if (region_counts != 1).any():
        raise ValueError("Conflicting anatomical regions for an image.")

    method_counts = validated.groupby("relative_path")["method"].nunique()

    if (method_counts != len(EXPECTED_METHODS)).any():
        raise ValueError("Every image must have both method evaluations.")

    if not validated["passed"].all():
        raise ValueError("Manifest contains failed method evaluations.")

    return validated.sort_values(
        ["body_region", "relative_path", "method"]
    ).reset_index(drop=True)


def bootstrap_mean_interval(
    values: np.ndarray,
    config: ComparisonConfig,
    seed_offset: int = 0,
) -> tuple[float, float]:
    """Calculate a descriptive image-level bootstrap interval."""

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1 or array.size == 0:
        raise ValueError("Bootstrap requires a nonempty 1D array.")

    if not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be finite.")

    rng = np.random.default_rng(config.random_seed + seed_offset)

    means = np.empty(
        config.bootstrap_iterations,
        dtype=np.float64,
    )

    for index in range(config.bootstrap_iterations):
        sample = rng.choice(
            array,
            size=array.size,
            replace=True,
        )

        means[index] = float(sample.mean())

    alpha = 1 - config.confidence_level

    lower, upper = np.quantile(
        means,
        [alpha / 2, 1 - alpha / 2],
    )

    return float(lower), float(upper)


def calculate_coverage_statistics(
    dataframe: pd.DataFrame,
    config: ComparisonConfig,
) -> pd.DataFrame:
    """Calculate descriptive coverage statistics by region."""

    records: list[dict] = []

    for region_index, region in enumerate(EXPECTED_REGIONS):
        for method_index, method in enumerate(EXPECTED_METHODS):
            subset = dataframe.loc[
                (dataframe["body_region"] == region) & (dataframe["method"] == method)
            ]

            if subset.empty:
                continue

            values = subset["coverage"].to_numpy(dtype=np.float64)

            lower, upper = bootstrap_mean_interval(
                values,
                config,
                seed_offset=region_index * 10 + method_index,
            )

            records.append(
                {
                    "body_region": region,
                    "method": method,
                    "image_count": len(values),
                    "mean_coverage": float(values.mean()),
                    "median_coverage": float(np.median(values)),
                    "std_coverage": float(values.std(ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "min_coverage": float(values.min()),
                    "max_coverage": float(values.max()),
                    "q25_coverage": float(np.quantile(values, 0.25)),
                    "q75_coverage": float(np.quantile(values, 0.75)),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "near_full_coverage_count": int(
                        (values >= config.near_full_coverage_threshold).sum()
                    ),
                }
            )

    return pd.DataFrame(records)


def calculate_paired_comparison(
    dataframe: pd.DataFrame,
    config: ComparisonConfig,
) -> pd.DataFrame:
    """Compare method coverage on the same images."""

    pivot = dataframe.pivot(
        index=["relative_path", "body_region"],
        columns="method",
        values="coverage",
    )

    if pivot[["ycrcb", "hsv"]].isna().any().any():
        raise ValueError("Paired comparison requires both methods.")

    paired = pivot.reset_index()

    paired["coverage_difference_hsv_minus_ycrcb"] = paired["hsv"] - paired["ycrcb"]

    paired["absolute_coverage_difference"] = paired[
        "coverage_difference_hsv_minus_ycrcb"
    ].abs()

    paired["large_disagreement"] = (
        paired["absolute_coverage_difference"] >= config.disagreement_threshold
    )

    paired["ycrcb_near_full"] = paired["ycrcb"] >= config.near_full_coverage_threshold

    paired["hsv_near_full"] = paired["hsv"] >= config.near_full_coverage_threshold

    return paired.sort_values(["body_region", "relative_path"]).reset_index(drop=True)


def create_comparison_summary(
    paired: pd.DataFrame,
    config: ComparisonConfig,
) -> dict:
    """Create research summary of paired coverage differences."""

    if paired.empty:
        raise ValueError("Cannot summarize an empty paired comparison.")

    differences = paired["coverage_difference_hsv_minus_ycrcb"].to_numpy(
        dtype=np.float64
    )

    lower, upper = bootstrap_mean_interval(
        differences,
        config,
        seed_offset=100,
    )

    region_summary = {}

    for region in EXPECTED_REGIONS:
        subset = paired[paired["body_region"] == region]

        if subset.empty:
            continue

        region_summary[region] = {
            "image_count": int(len(subset)),
            "mean_ycrcb_coverage": float(subset["ycrcb"].mean()),
            "mean_hsv_coverage": float(subset["hsv"].mean()),
            "mean_paired_difference": float(
                subset["coverage_difference_hsv_minus_ycrcb"].mean()
            ),
            "large_disagreement_count": int(subset["large_disagreement"].sum()),
        }

    return {
        "total_images": int(len(paired)),
        "methods": list(EXPECTED_METHODS),
        "mean_paired_difference_hsv_minus_ycrcb": float(differences.mean()),
        "mean_absolute_coverage_difference": float(np.abs(differences).mean()),
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "large_disagreement_count": int(paired["large_disagreement"].sum()),
        "ycrcb_near_full_count": int(paired["ycrcb_near_full"].sum()),
        "hsv_near_full_count": int(paired["hsv_near_full"].sum()),
        "region_summary": region_summary,
        "bootstrap_unit": "image",
        "patient_independence_verified": False,
        "segmentation_ground_truth_available": False,
        "clinical_validation": False,
    }

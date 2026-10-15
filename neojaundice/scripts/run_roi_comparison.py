"""Run reproducible ROI method coverage comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from ml.evaluation.roi_comparison import (
    ComparisonConfig,
    calculate_coverage_statistics,
    calculate_paired_comparison,
    create_comparison_summary,
    validate_manifest,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ROI segmentation coverage.")

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/roi_comparison.yaml"),
    )

    return parser.parse_args()


def load_config(path: Path) -> ComparisonConfig:
    """Load evaluation settings from YAML."""

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Invalid evaluation configuration.")

    if raw.get("schema_version") != "1.0":
        raise ValueError("Unsupported evaluation schema version.")

    statistics = raw.get("statistics", {})
    analysis = raw.get("analysis", {})

    return ComparisonConfig(
        bootstrap_iterations=statistics.get(
            "bootstrap_iterations",
            2000,
        ),
        confidence_level=statistics.get(
            "confidence_level",
            0.95,
        ),
        random_seed=statistics.get(
            "random_seed",
            42,
        ),
        near_full_coverage_threshold=analysis.get(
            "near_full_coverage_threshold",
            0.95,
        ),
        disagreement_threshold=analysis.get(
            "disagreement_threshold",
            0.05,
        ),
    )


def create_coverage_chart(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot coverage distributions by region and method."""

    groups = []
    labels = []

    for region in ("forehead", "sternum"):
        for method in ("ycrcb", "hsv"):
            values = dataframe.loc[
                (dataframe["body_region"] == region) & (dataframe["method"] == method),
                "coverage",
            ].to_numpy(dtype=float)

            if len(values):
                groups.append(values * 100)
                labels.append(f"{region.capitalize()}\n{method.upper()}")

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.boxplot(
        groups,
        tick_labels=labels,
        showmeans=True,
    )

    axis.set_title("Segmentation Coverage by Region and Method")

    axis.set_ylabel("Geometric ROI Coverage (%)")

    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def create_difference_chart(
    paired: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot paired HSV-minus-YCrCb coverage differences."""

    figure, axis = plt.subplots(figsize=(10, 6))

    for region in ("forehead", "sternum"):
        subset = paired[paired["body_region"] == region]

        values = (
            subset["coverage_difference_hsv_minus_ycrcb"].to_numpy(dtype=float) * 100
        )

        if len(values):
            axis.hist(
                values,
                bins=30,
                alpha=0.55,
                label=region.capitalize(),
            )

    axis.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axis.set_title("Paired Coverage Difference: HSV - YCrCb")

    axis.set_xlabel("Coverage Difference (percentage points)")

    axis.set_ylabel("Number of Images")

    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def main() -> int:
    args = parse_arguments()

    if not args.manifest.is_file():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        return 2

    try:
        config = load_config(args.config)

        raw = pd.read_csv(
            args.manifest,
            encoding="utf-8-sig",
        )

        dataframe = validate_manifest(raw)

        statistics = calculate_coverage_statistics(
            dataframe,
            config,
        )

        paired = calculate_paired_comparison(
            dataframe,
            config,
        )

        summary = create_comparison_summary(
            paired,
            config,
        )

        output_root = args.output.resolve()

        chart_directory = output_root / "charts"

        chart_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        statistics.to_csv(
            output_root / "coverage_statistics.csv",
            index=False,
        )

        paired.to_csv(
            output_root / "paired_comparison.csv",
            index=False,
        )

        create_coverage_chart(
            dataframe,
            chart_directory / "coverage_by_region.png",
        )

        create_difference_chart(
            paired,
            chart_directory / "coverage_difference.png",
        )

        summary["evaluation_config"] = {
            "bootstrap_iterations": (config.bootstrap_iterations),
            "confidence_level": (config.confidence_level),
            "random_seed": config.random_seed,
            "near_full_coverage_threshold": (config.near_full_coverage_threshold),
            "disagreement_threshold": (config.disagreement_threshold),
        }

        summary_path = output_root / "roi_comparison_summary.json"

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    except (
        ValueError,
        OSError,
        KeyError,
        yaml.YAMLError,
    ) as exc:
        print(f"[ERROR] ROI comparison failed: {exc}")
        return 2

    print("=" * 60)
    print("NeoJaundice — ROI Method Comparison")
    print("=" * 60)

    print(f"Images compared: {summary['total_images']}")

    print(
        "Mean paired difference (HSV - YCrCb): "
        f"{summary['mean_paired_difference_hsv_minus_ycrcb']:.6f}"
    )

    print(
        "Mean absolute coverage difference: "
        f"{summary['mean_absolute_coverage_difference']:.6f}"
    )

    print(f"Large disagreements: {summary['large_disagreement_count']}")

    print()
    print(f"Output directory: {output_root}")
    print("[PASSED] ROI comparison completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

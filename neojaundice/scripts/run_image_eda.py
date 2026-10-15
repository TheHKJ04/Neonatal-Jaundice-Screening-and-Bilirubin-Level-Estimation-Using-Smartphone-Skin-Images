"""Command-line interface for NeoJaundice image EDA."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ml.datasets.image_eda import (
    analyze_dataset,
    build_dataset_summary,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exploratory analysis on neonatal skin images."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def save_histogram(
    values,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """Save a single histogram."""

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.hist(values, bins=30)

    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Image count")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)

    plt.close(figure)


def generate_plots(
    dataframe,
    plots_directory: Path,
) -> None:
    """Generate reproducible descriptive plots."""

    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    configurations = [
        (
            "width",
            "Image Width Distribution",
            "Width (pixels)",
            "image_width_distribution.png",
        ),
        (
            "aspect_ratio",
            "Aspect Ratio Distribution",
            "Width / Height",
            "aspect_ratio_distribution.png",
        ),
        (
            "brightness",
            "Brightness Distribution",
            "Mean grayscale intensity",
            "brightness_distribution.png",
        ),
        (
            "contrast",
            "Contrast Distribution",
            "Grayscale standard deviation",
            "contrast_distribution.png",
        ),
        (
            "blur_score",
            "Laplacian Variance Distribution",
            "Laplacian variance",
            "blur_distribution.png",
        ),
        (
            "saturation_mean",
            "Saturation Distribution",
            "Mean HSV saturation",
            "saturation_distribution.png",
        ),
    ]

    for column, title, xlabel, filename in configurations:
        save_histogram(
            dataframe[column],
            title,
            xlabel,
            plots_directory / filename,
        )


def main() -> int:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.is_dir():
        LOGGER.error("Input dataset directory does not exist.")
        return 1

    if input_root == output_root or input_root in output_root.parents:
        LOGGER.error("Output directory must not be inside the input dataset.")
        return 1

    try:
        dataframe, failures = analyze_dataset(input_root)

        summary = build_dataset_summary(
            dataframe,
            failures,
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        csv_path = output_root / "image_statistics.csv"
        json_path = output_root / "dataset_summary.json"
        plots_directory = output_root / "plots"

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        json_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        generate_plots(
            dataframe,
            plots_directory,
        )

    except (ValueError, OSError) as exc:
        LOGGER.error(
            "EDA failed: %s",
            type(exc).__name__,
        )
        return 2

    print("=" * 60)
    print("NeoJaundice Image EDA")
    print("=" * 60)

    print(f"Images analyzed: {summary['analyzed_images']}")

    print(f"Failed images: {summary['failed_images']}")

    print(
        "Region folder counts:",
        summary["region_folder_counts"],
    )

    print()
    print(f"Statistics: {csv_path}")
    print(f"Summary: {json_path}")
    print(f"Plots: {plots_directory}")

    print()
    print("[PASSED] Image EDA completed.")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

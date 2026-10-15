"""CLI for forehead-versus-sternum exploratory analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ml.datasets.region_eda import (
    REGIONS,
    build_region_report,
    load_image_statistics,
    save_region_reports,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare forehead and sternum image statistics."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to image_statistics.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for region-specific reports",
    )

    return parser.parse_args()


def save_comparison_boxplot(
    dataframe,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a boxplot comparing the two anatomical regions."""

    values = [
        dataframe.loc[
            dataframe["body_region_folder"] == region,
            metric,
        ].to_numpy()
        for region in REGIONS
    ]

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.boxplot(
        values,
        tick_labels=["Forehead", "Sternum"],
        showfliers=True,
    )

    axis.set_title(title)
    axis.set_ylabel(ylabel)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def save_rgb_comparison(
    dataframe,
    output_path: Path,
) -> None:
    """Save grouped RGB channel mean comparisons."""

    channels = [
        "red_mean",
        "green_mean",
        "blue_mean",
    ]

    grouped = dataframe.groupby("body_region_folder")[channels].mean().reindex(REGIONS)

    figure, axis = plt.subplots(figsize=(8, 5))

    grouped.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title("RGB Channel Means by Anatomical Region")

    axis.set_xlabel("Body region")
    axis.set_ylabel("Mean channel intensity")
    axis.tick_params(
        axis="x",
        rotation=0,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def generate_region_plots(
    dataframe,
    plots_directory: Path,
) -> None:
    """Generate region-specific comparison figures."""

    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    configurations = [
        (
            "brightness",
            "Brightness by Anatomical Region",
            "Mean grayscale intensity",
            "brightness_comparison.png",
        ),
        (
            "contrast",
            "Contrast by Anatomical Region",
            "Grayscale standard deviation",
            "contrast_comparison.png",
        ),
        (
            "blur_score",
            "Laplacian Variance by Anatomical Region",
            "Laplacian variance",
            "blur_comparison.png",
        ),
        (
            "saturation_mean",
            "Saturation by Anatomical Region",
            "Mean HSV saturation",
            "saturation_comparison.png",
        ),
        (
            "overexposed_fraction",
            "Overexposure by Anatomical Region",
            "Fraction of grayscale pixels >= 245",
            "exposure_comparison.png",
        ),
    ]

    for metric, title, ylabel, filename in configurations:
        save_comparison_boxplot(
            dataframe=dataframe,
            metric=metric,
            title=title,
            ylabel=ylabel,
            output_path=plots_directory / filename,
        )

    save_rgb_comparison(
        dataframe,
        plots_directory / "rgb_comparison.png",
    )


def main() -> int:
    args = parse_arguments()

    try:
        dataframe = load_image_statistics(args.input)

        report, comparison = build_region_report(dataframe)

        csv_path, json_path = save_region_reports(
            report=report,
            comparison=comparison,
            output_directory=args.output,
        )

        plots_directory = args.output / "plots"

        generate_region_plots(
            dataframe,
            plots_directory,
        )

    except (ValueError, OSError) as exc:
        LOGGER.error(
            "Region EDA failed: %s",
            exc,
        )
        return 1

    print("=" * 60)
    print("NeoJaundice Region-Specific Image EDA")
    print("=" * 60)

    print(f"Total images: {report['total_images']}")

    for region in REGIONS:
        count = report["regions"][region]["image_count"]

        print(f"{region.capitalize()} images: {count}")

    print()
    print("Region comparison:")

    print(
        comparison[
            [
                "metric",
                "forehead_mean",
                "sternum_mean",
                "mean_difference",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")
    print(f"Plots: {plots_directory}")

    print()
    print("[PASSED] Region-specific EDA completed.")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

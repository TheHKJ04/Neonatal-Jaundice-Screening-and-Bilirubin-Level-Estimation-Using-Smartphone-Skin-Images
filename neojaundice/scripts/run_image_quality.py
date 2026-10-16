"""Run ROI image quality assessment across a dataset."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from ml.datasets.image_eda import (
    detect_body_region,
    discover_images,
)
from ml.evaluation.image_quality import (
    ImageQualityConfig,
    calculate_image_quality,
)
from ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    load_preprocessing_config,
)
from ml.roi.geometric_roi import (
    GeometricROIExtractor,
    load_roi_config,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run geometric ROI image quality assessment."
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

    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("configs/preprocessing/baseline.yaml"),
    )

    parser.add_argument(
        "--roi-config",
        type=Path,
        default=Path("configs/roi/geometric_baseline.yaml"),
    )

    parser.add_argument(
        "--quality-config",
        type=Path,
        default=Path("configs/evaluation/image_quality.yaml"),
    )

    return parser.parse_args()


def load_quality_config(
    path: Path,
) -> ImageQualityConfig:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Quality configuration must be a mapping.")

    if raw.get("schema_version") != "1.0":
        raise ValueError("Unsupported quality configuration schema.")

    brightness = raw["brightness"]
    warnings = raw["quality_warnings"]

    config = ImageQualityConfig(
        underexposed_pixel_threshold=(brightness["underexposed_pixel_threshold"]),
        overexposed_pixel_threshold=(brightness["overexposed_pixel_threshold"]),
        minimum_roi_pixels=(warnings["minimum_roi_pixels"]),
        low_brightness_threshold=(warnings["low_brightness_threshold"]),
        high_brightness_threshold=(warnings["high_brightness_threshold"]),
        low_contrast_threshold=(warnings["low_contrast_threshold"]),
        low_sharpness_threshold=(warnings["low_sharpness_threshold"]),
        maximum_underexposed_fraction=(warnings["maximum_underexposed_fraction"]),
        maximum_overexposed_fraction=(warnings["maximum_overexposed_fraction"]),
    )

    if not (
        0
        <= config.underexposed_pixel_threshold
        < config.overexposed_pixel_threshold
        <= 255
    ):
        raise ValueError("Invalid exposure pixel thresholds.")

    if config.minimum_roi_pixels <= 0:
        raise ValueError("minimum_roi_pixels must be positive.")

    if not (
        0 <= config.low_brightness_threshold <= 255
        and 0 <= config.high_brightness_threshold <= 255
        and config.low_brightness_threshold < config.high_brightness_threshold
    ):
        raise ValueError("Invalid brightness warning thresholds.")

    if config.low_contrast_threshold < 0 or config.low_sharpness_threshold < 0:
        raise ValueError("Contrast and sharpness thresholds cannot be negative.")

    if not (
        0 <= config.maximum_underexposed_fraction <= 1
        and 0 <= config.maximum_overexposed_fraction <= 1
    ):
        raise ValueError("Exposure fractions must be between 0 and 1.")

    return config


def create_summary(
    dataframe: pd.DataFrame,
    image_count: int,
    failure_counts: Counter,
) -> dict:
    summary = {
        "total_images": image_count,
        "processed_images": int(len(dataframe)),
        "processing_failures": dict(failure_counts),
        "warning_counts": {},
        "region_summary": {},
        "clinical_validation": False,
        "quality_thresholds_are_exploratory": True,
    }

    for warning_string in dataframe["warnings"]:
        if warning_string:
            for warning in warning_string.split(";"):
                summary["warning_counts"][warning] = (
                    summary["warning_counts"].get(
                        warning,
                        0,
                    )
                    + 1
                )

    for region in ("forehead", "sternum"):
        subset = dataframe[dataframe["body_region"] == region]

        summary["region_summary"][region] = {
            "image_count": int(len(subset)),
            "mean_brightness": (
                float(subset["mean_brightness"].mean()) if not subset.empty else None
            ),
            "mean_contrast": (
                float(subset["contrast"].mean()) if not subset.empty else None
            ),
            "mean_laplacian_variance": (
                float(subset["laplacian_variance"].mean()) if not subset.empty else None
            ),
            "mean_underexposed_fraction": (
                float(subset["underexposed_fraction"].mean())
                if not subset.empty
                else None
            ),
            "mean_overexposed_fraction": (
                float(subset["overexposed_fraction"].mean())
                if not subset.empty
                else None
            ),
        }

    summary["status"] = (
        "passed" if not failure_counts and len(dataframe) == image_count else "failed"
    )

    return summary


def main() -> int:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.is_dir():
        print("[ERROR] Input directory does not exist.")
        return 2

    if input_root == output_root or input_root in output_root.parents:
        print("[ERROR] Output cannot be inside raw dataset.")
        return 2

    try:
        preprocessing_config = load_preprocessing_config(args.preprocessing_config)

        roi_config = load_roi_config(args.roi_config)

        quality_config = load_quality_config(args.quality_config)

        preprocessor = ImagePreprocessor(preprocessing_config)

        roi_extractor = GeometricROIExtractor(roi_config)

        image_paths = sorted(discover_images(input_root))

        if not image_paths:
            raise ValueError("No supported images found.")

        records = []
        failure_counts = Counter()

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            relative_path = image_path.relative_to(input_root).as_posix()

            try:
                region = detect_body_region(
                    image_path,
                    input_root,
                )

                if region not in (
                    "forehead",
                    "sternum",
                ):
                    raise ValueError("Unknown anatomical region.")

                preprocessed = preprocessor.process(image_path)

                roi = roi_extractor.extract(
                    preprocessed,
                    region,
                )

                result = calculate_image_quality(
                    preprocessed.preview_rgb,
                    roi.roi_mask,
                    quality_config,
                )

                record = {
                    "relative_path": relative_path,
                    "body_region": region,
                    **asdict(result),
                }

                record["warnings"] = ";".join(result.warnings)

                record["warning_count"] = len(result.warnings)

                records.append(record)

            except (
                ValueError,
                OSError,
                TypeError,
            ) as exc:
                failure_counts[type(exc).__name__] += 1

                LOGGER.warning(
                    "Failed image %s: %s",
                    relative_path,
                    type(exc).__name__,
                )

            if index % 100 == 0:
                LOGGER.info(
                    "Processed %s/%s images",
                    index,
                    len(image_paths),
                )

        dataframe = pd.DataFrame(records)

        if dataframe.empty:
            raise ValueError("No successful image-quality evaluations.")

        manifest_path = output_root / "image_quality_manifest.csv"

        dataframe.to_csv(
            manifest_path,
            index=False,
        )

        review_candidates = dataframe[dataframe["warning_count"] > 0].copy()

        review_candidates = review_candidates.sort_values(
            by=[
                "warning_count",
                "relative_path",
            ],
            ascending=[
                False,
                True,
            ],
        )

        review_path = output_root / "quality_review_candidates.csv"

        review_candidates.to_csv(
            review_path,
            index=False,
        )

        summary = create_summary(
            dataframe,
            len(image_paths),
            failure_counts,
        )

        summary_path = output_root / "image_quality_summary.json"

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
        TypeError,
        KeyError,
        yaml.YAMLError,
    ) as exc:
        LOGGER.error(
            "Image-quality experiment failed: %s",
            exc,
        )
        return 2

    print("=" * 60)
    print("NeoJaundice — Phase 5.6B")
    print("Geometric ROI Image Quality Assessment")
    print("=" * 60)

    print(f"Images: {summary['total_images']}")

    print(f"Processed: {summary['processed_images']}")

    print(f"Review candidates: {len(review_candidates)}")

    print(f"Status: {summary['status']}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Review candidates: {review_path}")

    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

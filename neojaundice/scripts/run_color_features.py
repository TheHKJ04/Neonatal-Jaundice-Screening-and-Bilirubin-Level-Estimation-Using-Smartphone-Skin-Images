"""Extract reproducible color features from neonatal skin ROIs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from ml.datasets.image_eda import (
    detect_body_region,
    discover_images,
)
from ml.features.color_features import extract_color_features
from ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    load_preprocessing_config,
)
from ml.roi.geometric_roi import (
    GeometricROIExtractor,
    load_roi_config,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract color features from geometric ROIs."
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

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
        "--feature-config",
        type=Path,
        default=Path("configs/features/color_baseline.yaml"),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.is_dir():
        LOGGER.error("Input directory does not exist.")
        return 2

    if input_root == output_root or input_root in output_root.parents:
        LOGGER.error("Output cannot be inside the raw dataset.")
        return 2

    try:
        with args.feature_config.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            feature_config = yaml.safe_load(file)

        if not isinstance(feature_config, dict):
            raise ValueError("Invalid feature configuration.")

        if feature_config.get("schema_version") != "1.0":
            raise ValueError("Unsupported feature schema.")

        expected_spaces = {"rgb", "hsv", "ycrcb", "lab"}
        expected_statistics = {"mean", "std", "median", "p10", "p90"}

        configured_features = feature_config["features"]

        if set(configured_features["color_spaces"]) != expected_spaces:
            raise ValueError("Unexpected color-space configuration.")

        if set(configured_features["statistics"]) != expected_statistics:
            raise ValueError("Unexpected statistics configuration.")

        preprocessor = ImagePreprocessor(
            load_preprocessing_config(args.preprocessing_config)
        )

        roi_extractor = GeometricROIExtractor(load_roi_config(args.roi_config))

        image_paths = sorted(discover_images(input_root))

        if not image_paths:
            raise ValueError("No images found.")

        records = []
        failures = {}

        for index, image_path in enumerate(image_paths, start=1):
            relative_path = image_path.relative_to(input_root).as_posix()

            try:
                region = detect_body_region(image_path, input_root)

                if region not in {"forehead", "sternum"}:
                    raise ValueError("Unsupported body region.")

                preprocessed = preprocessor.process(image_path)

                roi = roi_extractor.extract(preprocessed, region)

                features = extract_color_features(
                    preprocessed.preview_rgb,
                    roi.roi_mask,
                )

                records.append(
                    {
                        "relative_path": relative_path,
                        "body_region": region,
                        **features,
                    }
                )

            except (ValueError, TypeError, OSError, cv2.error) as exc:
                failures[relative_path] = type(exc).__name__
                LOGGER.warning(
                    "Failed: %s (%s)",
                    relative_path,
                    type(exc).__name__,
                )

            if index % 100 == 0:
                LOGGER.info(
                    "Processed %s/%s",
                    index,
                    len(image_paths),
                )

        if not records:
            raise ValueError("No successful feature extractions.")

        dataframe = pd.DataFrame(records)

        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns

        if not np.isfinite(dataframe[numeric_columns].to_numpy()).all():
            raise ValueError("Non-finite feature values detected.")

        output_root.mkdir(parents=True, exist_ok=True)

        manifest_path = output_root / "color_features.csv"
        dataframe.to_csv(manifest_path, index=False)

        summary = {
            "total_images": len(image_paths),
            "processed_images": len(dataframe),
            "processing_failures": failures,
            "feature_count": len(numeric_columns),
            "region_counts": {
                str(key): int(value)
                for key, value in dataframe["body_region"].value_counts().items()
            },
            "clinical_validation": False,
            "bilirubin_regression_trained": False,
            "status": (
                "passed"
                if len(dataframe) == len(image_paths) and not failures
                else "failed"
            ),
        }

        summary_path = output_root / "color_features_summary.json"

        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("=" * 55)
        print("NeoJaundice — Phase 6.1")
        print("=" * 55)
        print(f"Images: {summary['total_images']}")
        print(f"Processed: {summary['processed_images']}")
        print(f"Numerical features: {summary['feature_count']}")
        print(f"Status: {summary['status']}")
        print(f"Manifest: {manifest_path}")
        print(f"Summary: {summary_path}")

        return 0 if summary["status"] == "passed" else 1

    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        LOGGER.error("Feature extraction failed: %s", exc)
        return 2


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

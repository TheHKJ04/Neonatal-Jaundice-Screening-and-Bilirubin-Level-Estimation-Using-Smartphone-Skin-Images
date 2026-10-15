"""Batch experimental skin segmentation and comparison."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.datasets.image_eda import (
    detect_body_region,
    discover_images,
)
from ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    load_preprocessing_config,
)
from ml.roi.geometric_roi import (
    GeometricROIExtractor,
    create_roi_overlay,
    load_roi_config,
)
from ml.roi.segmentation_quality import (
    validate_segmentation_result,
)
from ml.roi.skin_segmentation import (
    SUPPORTED_METHODS,
    SkinSegmenter,
    create_segmentation_overlay,
    load_segmentation_config,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimental skin segmentation.")

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
        "--segmentation-config",
        type=Path,
        default=Path("configs/roi/skin_segmentation.yaml"),
    )

    parser.add_argument(
        "--previews-per-region",
        type=int,
        default=5,
    )

    return parser.parse_args()


def save_comparison_preview(
    preview_rgb: np.ndarray,
    roi_overlay: np.ndarray,
    ycrcb_overlay: np.ndarray,
    hsv_overlay: np.ndarray,
    output_path: Path,
) -> None:
    """Save four images side by side."""

    comparison = np.concatenate(
        [
            preview_rgb,
            roi_overlay,
            ycrcb_overlay,
            hsv_overlay,
        ],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(comparison).save(output_path)


def main() -> int:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if args.previews_per_region < 0:
        LOGGER.error("Preview count cannot be negative.")
        return 2

    if input_root == output_root or input_root in output_root.parents:
        LOGGER.error("Output cannot be inside the raw dataset.")
        return 2

    if not input_root.is_dir():
        LOGGER.error("Input directory does not exist.")
        return 2

    try:
        preprocessing_config = load_preprocessing_config(args.preprocessing_config)

        roi_config = load_roi_config(args.roi_config)

        segmentation_config = load_segmentation_config(args.segmentation_config)

        preprocessor = ImagePreprocessor(preprocessing_config)

        roi_extractor = GeometricROIExtractor(roi_config)

        segmenter = SkinSegmenter(segmentation_config)

        image_paths = sorted(discover_images(input_root))

        if not image_paths:
            raise ValueError("No supported images found.")

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = []

        preview_counts = {
            "forehead": 0,
            "sternum": 0,
        }

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            relative_path = image_path.relative_to(input_root).as_posix()

            region = detect_body_region(
                image_path,
                input_root,
            )

            try:
                if region not in (
                    "forehead",
                    "sternum",
                ):
                    raise ValueError("Unsupported anatomical region.")

                preprocessed = preprocessor.process(image_path)

                roi = roi_extractor.extract(
                    preprocessed,
                    region,
                )

                results = {}

                for method in SUPPORTED_METHODS:
                    result = segmenter.segment(
                        preprocessed.preview_rgb,
                        roi.roi_mask,
                        method,
                    )

                    qc = validate_segmentation_result(
                        result,
                        roi.roi_mask,
                    )

                    results[method] = (
                        result,
                        qc,
                    )

                    records.append(
                        {
                            "relative_path": relative_path,
                            "body_region": region,
                            "method": method,
                            "passed": qc.passed,
                            "errors": ";".join(qc.errors),
                            "selected_pixel_count": (qc.selected_pixel_count),
                            "geometric_roi_pixel_count": (
                                result.geometric_roi_pixel_count
                            ),
                            "coverage": qc.coverage,
                            "coverage_warning": (result.coverage_warning or ""),
                        }
                    )

                if preview_counts[region] < args.previews_per_region:
                    roi_overlay = create_roi_overlay(
                        preprocessed.preview_rgb,
                        roi,
                    )

                    ycrcb_overlay = create_segmentation_overlay(
                        preprocessed.preview_rgb,
                        results["ycrcb"][0],
                    )

                    hsv_overlay = create_segmentation_overlay(
                        preprocessed.preview_rgb,
                        results["hsv"][0],
                    )

                    preview_directory = output_root / "previews" / region

                    preview_number = preview_counts[region] + 1

                    preview_path = (
                        preview_directory / f"comparison_{preview_number:03d}.png"
                    )

                    save_comparison_preview(
                        preprocessed.preview_rgb,
                        roi_overlay,
                        ycrcb_overlay,
                        hsv_overlay,
                        preview_path,
                    )

                    preview_counts[region] += 1

            except (ValueError, OSError) as exc:
                LOGGER.warning(
                    "Segmentation failed for image %s: %s",
                    index,
                    type(exc).__name__,
                )

                for method in SUPPORTED_METHODS:
                    records.append(
                        {
                            "relative_path": relative_path,
                            "body_region": region,
                            "method": method,
                            "passed": False,
                            "errors": type(exc).__name__,
                            "selected_pixel_count": 0,
                            "geometric_roi_pixel_count": 0,
                            "coverage": 0.0,
                            "coverage_warning": "",
                        }
                    )

            if index % 100 == 0:
                LOGGER.info(
                    "Processed %s/%s images",
                    index,
                    len(image_paths),
                )

        dataframe = pd.DataFrame(records)

        manifest_path = output_root / "segmentation_manifest.csv"

        dataframe.to_csv(
            manifest_path,
            index=False,
        )

        method_summary = {}

        for method in SUPPORTED_METHODS:
            subset = dataframe[dataframe["method"] == method]

            passed_subset = subset[subset["passed"]]

            method_summary[method] = {
                "total_images": len(subset),
                "passed_images": len(passed_subset),
                "failed_images": (len(subset) - len(passed_subset)),
                "mean_coverage": (
                    float(passed_subset["coverage"].mean())
                    if not passed_subset.empty
                    else None
                ),
                "low_coverage_warnings": int(
                    (subset["coverage_warning"] == "low_coverage").sum()
                ),
                "high_coverage_warnings": int(
                    (subset["coverage_warning"] == "high_coverage").sum()
                ),
            }

        total_failed = int((~dataframe["passed"]).sum())

        summary = {
            "total_images": len(image_paths),
            "total_method_evaluations": (len(dataframe)),
            "failed_method_evaluations": (total_failed),
            "region_counts": {
                region: sum(
                    detect_body_region(
                        path,
                        input_root,
                    )
                    == region
                    for path in image_paths
                )
                for region in (
                    "forehead",
                    "sternum",
                )
            },
            "methods": method_summary,
            "preview_counts": preview_counts,
            "status": ("passed" if total_failed == 0 else "failed"),
            "clinical_validation": False,
            "segmentation_ground_truth_available": False,
        }

        summary_path = output_root / "segmentation_summary.json"

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
    ) as exc:
        LOGGER.error(
            "Segmentation pipeline failed: %s",
            type(exc).__name__,
        )
        return 2

    print("=" * 60)
    print("NeoJaundice Experimental Skin Segmentation")
    print("=" * 60)

    print(f"Images: {summary['total_images']}")

    print(f"Method evaluations: {summary['total_method_evaluations']}")

    print(f"Failed evaluations: {summary['failed_method_evaluations']}")

    print(f"Status: {summary['status']}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    if total_failed > 0:
        print("[FAILED] Segmentation QC failed.")
        return 1

    print("[PASSED] Segmentation QC completed.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

"""Batch geometric ROI extraction for NeoJaundice."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geometric ROI extraction.")

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
        "--previews-per-region",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if args.previews_per_region < 0:
        LOGGER.error("Preview count must be nonnegative.")
        return 2

    if input_root == output_root or input_root in output_root.parents:
        LOGGER.error("Output cannot be inside the raw dataset.")
        return 2

    try:
        preprocessing_config = load_preprocessing_config(args.preprocessing_config)

        roi_config = load_roi_config(args.roi_config)

        preprocessor = ImagePreprocessor(preprocessing_config)

        extractor = GeometricROIExtractor(roi_config)

        image_paths = discover_images(input_root)

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
                preprocessing_result = preprocessor.process(image_path)

                roi_result = extractor.extract(
                    preprocessing_result,
                    region=region,
                )

                x_min, y_min, x_max, y_max = roi_result.pixel_box

                records.append(
                    {
                        "relative_path": relative_path,
                        "body_region": region,
                        "passed": True,
                        "error": "",
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "roi_pixel_count": (roi_result.roi_pixel_count),
                        "valid_pixel_count": (roi_result.valid_pixel_count),
                        "roi_fraction_of_valid_image": (
                            roi_result.roi_fraction_of_valid_image
                        ),
                        "roi_fraction_of_canvas": (roi_result.roi_fraction_of_canvas),
                    }
                )

                if (
                    region in preview_counts
                    and preview_counts[region] < args.previews_per_region
                ):
                    overlay = create_roi_overlay(
                        preprocessing_result.preview_rgb,
                        roi_result,
                    )

                    preview_directory = output_root / "previews" / region

                    preview_directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    preview_number = preview_counts[region] + 1

                    preview_path = preview_directory / f"roi_{preview_number:03d}.png"

                    Image.fromarray(overlay).save(preview_path)

                    preview_counts[region] += 1

            except (ValueError, OSError) as exc:
                LOGGER.warning(
                    "ROI failed for image %s: %s",
                    index,
                    type(exc).__name__,
                )

                records.append(
                    {
                        "relative_path": relative_path,
                        "body_region": region,
                        "passed": False,
                        "error": type(exc).__name__,
                    }
                )

            if index % 100 == 0:
                LOGGER.info(
                    "Processed %s/%s images",
                    index,
                    len(image_paths),
                )

        dataframe = pd.DataFrame(records)

        manifest_path = output_root / "roi_manifest.csv"

        dataframe.to_csv(
            manifest_path,
            index=False,
        )

        passed = int(dataframe["passed"].sum())

        failed = len(dataframe) - passed

        summary = {
            "total_images": len(dataframe),
            "passed_images": passed,
            "failed_images": failed,
            "region_counts": (dataframe["body_region"].value_counts().to_dict()),
            "preview_counts": preview_counts,
            "status": ("passed" if failed == 0 else "failed"),
            "roi_method": "geometric_baseline",
            "clinical_roi_validation": False,
        }

        summary_path = output_root / "roi_summary.json"

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    except (ValueError, OSError) as exc:
        LOGGER.error(
            "ROI pipeline failed: %s",
            type(exc).__name__,
        )
        return 2

    print("=" * 60)
    print("NeoJaundice Geometric ROI Extraction")
    print("=" * 60)

    print(f"Total images: {summary['total_images']}")

    print(f"Passed images: {summary['passed_images']}")

    print(f"Failed images: {summary['failed_images']}")

    print(f"Region counts: {summary['region_counts']}")

    print(f"Preview counts: {summary['preview_counts']}")

    print(f"Status: {summary['status']}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    if failed > 0:
        print("[FAILED] ROI extraction had failures.")
        return 1

    print("[PASSED] ROI extraction completed.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

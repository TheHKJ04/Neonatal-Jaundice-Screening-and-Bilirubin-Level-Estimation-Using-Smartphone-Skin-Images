"""Batch quality-control CLI for image preprocessing."""

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
from ml.preprocessing.quality_control import (
    summarize_qc_results,
    validate_preprocessing_result,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing quality-control checks."
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
        "--config",
        type=Path,
        default=Path("configs/preprocessing/baseline.yaml"),
    )

    parser.add_argument(
        "--previews-per-region",
        type=int,
        default=3,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if args.previews_per_region < 0:
        LOGGER.error("previews-per-region must be nonnegative.")
        return 2

    if input_root == output_root or input_root in output_root.parents:
        LOGGER.error("Output must not be inside the raw dataset.")
        return 2

    try:
        config = load_preprocessing_config(args.config)

        image_paths = discover_images(input_root)

        if not image_paths:
            raise ValueError("No supported images found.")

        preprocessor = ImagePreprocessor(config)

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
            region = detect_body_region(
                image_path,
                input_root,
            )

            relative_path = image_path.relative_to(input_root).as_posix()

            try:
                result = preprocessor.process(image_path)

                qc = validate_preprocessing_result(
                    result,
                    config,
                )

                records.append(
                    {
                        "relative_path": relative_path,
                        "body_region": region,
                        "passed": qc.passed,
                        "errors": ";".join(qc.errors),
                        "valid_pixel_fraction": (qc.valid_pixel_fraction),
                    }
                )

                if (
                    qc.passed
                    and region in preview_counts
                    and preview_counts[region] < args.previews_per_region
                ):
                    preview_directory = output_root / "previews" / region

                    preview_directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    preview_number = preview_counts[region] + 1

                    preview_path = (
                        preview_directory / f"preview_{preview_number:03d}.png"
                    )

                    Image.fromarray(result.preview_rgb).save(preview_path)

                    preview_counts[region] += 1

            except (ValueError, OSError) as exc:
                LOGGER.warning(
                    "QC failed for image index %s: %s",
                    index,
                    type(exc).__name__,
                )

                records.append(
                    {
                        "relative_path": relative_path,
                        "body_region": region,
                        "passed": False,
                        "errors": type(exc).__name__,
                        "valid_pixel_fraction": None,
                    }
                )

            if index % 100 == 0:
                LOGGER.info(
                    "Checked %s/%s images",
                    index,
                    len(image_paths),
                )

        dataframe = pd.DataFrame(records)

        csv_path = output_root / "qc_results.csv"

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        summary = summarize_qc_results(records)

        summary["preview_counts"] = preview_counts

        summary_path = output_root / "qc_summary.json"

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    except (ValueError, OSError) as exc:
        LOGGER.error(
            "QC pipeline failed: %s",
            type(exc).__name__,
        )
        return 2

    print("=" * 60)
    print("NeoJaundice Preprocessing Quality Control")
    print("=" * 60)

    print(f"Total images: {summary['total_images']}")

    print(f"Passed images: {summary['passed_images']}")

    print(f"Failed images: {summary['failed_images']}")

    print(f"Region counts: {summary['region_counts']}")

    print(f"Status: {summary['status']}")

    print()
    print(f"CSV report: {csv_path}")
    print(f"JSON summary: {summary_path}")

    if summary["status"] != "passed":
        print("[FAILED] Preprocessing QC detected failures.")
        return 1

    print("[PASSED] Preprocessing QC completed.")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

"""Batch preprocessing validation for NeoJaundice."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
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

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate neonatal image preprocessing."
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

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if input_root == output_root or input_root in output_root.parents:
        LOGGER.error("Output must not be inside the raw input dataset.")
        return 1

    try:
        config = load_preprocessing_config(args.config)

        preprocessor = ImagePreprocessor(config)

        image_paths = discover_images(input_root)

        if not image_paths:
            raise ValueError("No supported images found.")

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        preview_directory = output_root / "previews"

        preview_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = []
        failures = []

        preview_limit = 6

        # The baseline YAML currently specifies six previews.
        # A configurable preview loader will be added below.

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            try:
                result = preprocessor.process(image_path)

                expected_image_shape = (
                    3,
                    config.target_height,
                    config.target_width,
                )

                expected_mask_shape = (
                    1,
                    config.target_height,
                    config.target_width,
                )

                if tuple(result.image_tensor.shape) != (expected_image_shape):
                    raise ValueError("Unexpected image tensor shape.")

                if tuple(result.valid_mask.shape) != (expected_mask_shape):
                    raise ValueError("Unexpected valid mask shape.")

                relative_path = image_path.relative_to(input_root).as_posix()

                region = detect_body_region(
                    image_path,
                    input_root,
                )

                record = {
                    "relative_path": relative_path,
                    "body_region": region,
                    **asdict(result.metadata),
                    "valid_pixel_fraction": float(result.valid_mask.mean().item()),
                }

                records.append(record)

                if len(records) <= preview_limit:
                    preview_name = f"preview_{len(records):03d}.png"

                    Image.fromarray(
                        result.preview_rgb,
                        mode="RGB",
                    ).save(preview_directory / preview_name)

            except (ValueError, OSError) as exc:
                failures.append(
                    {
                        "image_index": index,
                        "error_type": type(exc).__name__,
                    }
                )

                LOGGER.warning(
                    "Preprocessing failed at image index %s",
                    index,
                )

            if index % 100 == 0:
                LOGGER.info(
                    "Processed %s/%s images",
                    index,
                    len(image_paths),
                )

        if not records:
            raise ValueError("No images were processed successfully.")

        manifest_path = output_root / "manifest.csv"

        pd.DataFrame(records).to_csv(
            manifest_path,
            index=False,
        )

        summary = {
            "total_images": len(image_paths),
            "processed_images": len(records),
            "failed_images": len(failures),
            "target_width": config.target_width,
            "target_height": config.target_height,
            "normalization": config.normalization,
            "region_counts": (
                pd.DataFrame(records)["body_region"].value_counts().to_dict()
            ),
            "failures": failures,
            "clinical_labels_available": False,
        }

        summary_path = output_root / "summary.json"

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    except (ValueError, OSError) as exc:
        LOGGER.error(
            "Preprocessing failed: %s",
            type(exc).__name__,
        )
        return 2

    print("=" * 60)
    print("NeoJaundice Image Preprocessing")
    print("=" * 60)

    print(f"Total images: {summary['total_images']}")

    print(f"Processed images: {summary['processed_images']}")

    print(f"Failed images: {summary['failed_images']}")

    print(f"Region counts: {summary['region_counts']}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    print()
    if summary["failed_images"] > 0:
        print("[FAILED] Some images could not be preprocessed.")
        return 1

    print("[PASSED] Preprocessing validation completed.")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

"""Run technical ROI quality control and prepare visual review."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
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
from ml.roi.quality_control import (
    select_review_samples,
    summarize_roi_qc,
    validate_roi_result,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ROI quality control.")

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
        "--qc-config",
        type=Path,
        default=Path("configs/roi/quality_control.yaml"),
    )

    return parser.parse_args()


def load_qc_settings(path: Path) -> tuple[int, int]:
    """Load reproducible review sampling settings."""

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid QC configuration.")

    if config.get("schema_version") != "1.0":
        raise ValueError("Unsupported QC schema version.")

    sampling = config.get("sampling")

    if not isinstance(sampling, dict):
        raise ValueError("Missing sampling configuration.")

    if sampling.get("method") != "region_stratified_random":
        raise ValueError("Unsupported sampling method.")

    count = sampling.get("samples_per_region")
    seed = sampling.get("random_seed")

    if type(count) is not int or count < 0 or type(seed) is not int:
        raise ValueError("Invalid sampling count or random seed.")

    return count, seed


def main() -> int:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if input_root == output_root or input_root in output_root.parents:
        LOGGER.error("Output cannot be inside the input dataset.")
        return 2

    if not input_root.is_dir():
        LOGGER.error("Input directory does not exist.")
        return 2

    try:
        preprocessing_config = load_preprocessing_config(args.preprocessing_config)

        roi_config = load_roi_config(args.roi_config)

        samples_per_region, random_seed = load_qc_settings(args.qc_config)

        preprocessor = ImagePreprocessor(preprocessing_config)

        extractor = GeometricROIExtractor(roi_config)

        image_paths = sorted(discover_images(input_root))

        if not image_paths:
            raise ValueError("No images found.")

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        records: list[dict] = []

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
                if region not in ("forehead", "sternum"):
                    raise ValueError("Unsupported anatomical region.")

                preprocessed = preprocessor.process(image_path)

                roi = extractor.extract(
                    preprocessed,
                    region,
                )

                qc = validate_roi_result(
                    preprocessed,
                    roi,
                )

                records.append(
                    {
                        "relative_path": relative_path,
                        "body_region": region,
                        "passed": qc.passed,
                        "errors": ";".join(qc.errors),
                        "roi_pixel_count": (qc.roi_pixel_count),
                        "valid_pixel_count": (qc.valid_pixel_count),
                        "roi_fraction_of_valid_image": (qc.roi_fraction_of_valid_image),
                    }
                )

            except (ValueError, OSError) as exc:
                LOGGER.warning(
                    "ROI QC failed at image %s: %s",
                    index,
                    type(exc).__name__,
                )

                records.append(
                    {
                        "relative_path": relative_path,
                        "body_region": region,
                        "passed": False,
                        "errors": type(exc).__name__,
                        "roi_pixel_count": 0,
                        "valid_pixel_count": 0,
                        "roi_fraction_of_valid_image": 0.0,
                    }
                )

            if index % 100 == 0:
                LOGGER.info(
                    "Validated %s/%s images",
                    index,
                    len(image_paths),
                )

        qc_dataframe = pd.DataFrame(records)

        qc_csv = output_root / "qc_results.csv"

        qc_dataframe.to_csv(
            qc_csv,
            index=False,
        )

        summary = summarize_roi_qc(records)

        selected = select_review_samples(
            records,
            samples_per_region=samples_per_region,
            random_seed=random_seed,
        )

        review_records = []

        for index, record in enumerate(
            selected,
            start=1,
        ):
            relative_path = record["relative_path"]
            region = record["body_region"]

            image_path = input_root / Path(relative_path)

            preprocessed = preprocessor.process(image_path)

            roi = extractor.extract(
                preprocessed,
                region,
            )

            overlay = create_roi_overlay(
                preprocessed.preview_rgb,
                roi,
            )

            preview_directory = output_root / "review_previews" / region

            preview_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            preview_filename = f"review_{index:03d}.png"

            preview_path = preview_directory / preview_filename

            Image.fromarray(overlay).save(preview_path)

            review_records.append(
                {
                    "review_id": index,
                    "relative_path": relative_path,
                    "body_region": region,
                    "preview_path": (preview_path.relative_to(output_root).as_posix()),
                    "review_status": "pending",
                    "skin_region_visible": "",
                    "hair_or_eyebrow_contamination": "",
                    "clothing_contamination": "",
                    "background_contamination": "",
                    "strong_shadow_or_reflection": "",
                    "reviewer_notes": "",
                }
            )

        review_manifest = output_root / "review_manifest.csv"

        pd.DataFrame(
            review_records,
            columns=[
                "review_id",
                "relative_path",
                "body_region",
                "preview_path",
                "review_status",
                "skin_region_visible",
                "hair_or_eyebrow_contamination",
                "clothing_contamination",
                "background_contamination",
                "strong_shadow_or_reflection",
                "reviewer_notes",
            ],
        ).to_csv(
            review_manifest,
            index=False,
        )

        summary["sampling"] = {
            "method": "region_stratified_random",
            "random_seed": random_seed,
            "requested_samples_per_region": (samples_per_region),
            "selected_images": len(selected),
        }

        summary["visual_review_status"] = "pending"

        summary_path = output_root / "qc_summary.json"

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
        LOGGER.error(
            "ROI QC pipeline failed: %s",
            type(exc).__name__,
        )
        return 2

    print("=" * 60)
    print("NeoJaundice ROI Quality Control")
    print("=" * 60)

    print(f"Total images: {summary['total_images']}")

    print(f"Passed images: {summary['passed_images']}")

    print(f"Failed images: {summary['failed_images']}")

    print(f"Technical QC status: {summary['status']}")

    print(f"Review samples: {len(selected)}")

    print("Visual review status: pending")

    print()
    print(f"QC report: {qc_csv}")
    print(f"QC summary: {summary_path}")
    print(f"Review manifest: {review_manifest}")

    if summary["status"] != "passed":
        print("[FAILED] ROI technical QC failed.")
        return 1

    print("[PASSED] ROI technical QC completed.")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

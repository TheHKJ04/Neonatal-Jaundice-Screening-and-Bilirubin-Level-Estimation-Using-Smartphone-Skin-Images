"""Run pixel-level agreement and threshold sensitivity experiments."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from ml.datasets.image_eda import (
    detect_body_region,
    discover_images,
)
from ml.evaluation.mask_agreement import (
    calculate_mask_agreement,
    create_agreement_overlay,
)
from ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    load_preprocessing_config,
)
from ml.roi.geometric_roi import (
    GeometricROIExtractor,
    load_roi_config,
)
from ml.roi.segmentation_quality import (
    validate_segmentation_result,
)
from ml.roi.skin_segmentation import (
    ColorThreshold,
    SegmentationConfig,
    SkinSegmenter,
    load_segmentation_config,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pixel-level mask agreement and threshold sensitivity analysis."
        )
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
        "--experiment-config",
        type=Path,
        default=Path("configs/evaluation/mask_agreement.yaml"),
    )

    return parser.parse_args()


def load_experiment(
    path: Path,
) -> tuple[dict[str, SegmentationConfig], int]:
    """Load threshold variants and preview settings."""

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Experiment configuration must be a mapping.")

    if raw.get("schema_version") != "1.0":
        raise ValueError("Unsupported experiment schema version.")

    baseline_path = Path(raw["baseline_segmentation_config"])

    baseline = load_segmentation_config(baseline_path)

    variants = {
        "baseline": baseline,
    }

    variant_settings = raw["variants"]

    if not isinstance(variant_settings, dict):
        raise ValueError("Variants must be a mapping.")

    if set(variant_settings) != {
        "baseline",
        "moderate",
        "restrictive",
    }:
        raise ValueError("Expected baseline, moderate and restrictive variants.")

    if variant_settings["baseline"] != {"use_baseline_config": True}:
        raise ValueError("Baseline variant must use the original configuration.")

    for variant_name in (
        "moderate",
        "restrictive",
    ):
        settings = variant_settings[variant_name]

        if not isinstance(settings, dict):
            raise ValueError(f"Invalid variant: {variant_name}")

        method_thresholds = {}

        for method in ("ycrcb", "hsv"):
            threshold_settings = settings[method]

            method_thresholds[method] = ColorThreshold(
                lower=tuple(threshold_settings["lower"]),
                upper=tuple(threshold_settings["upper"]),
            )

        variants[variant_name] = SegmentationConfig(
            ycrcb=method_thresholds["ycrcb"],
            hsv=method_thresholds["hsv"],
            low_coverage_warning=(baseline.low_coverage_warning),
            high_coverage_warning=(baseline.high_coverage_warning),
        )

    previews_per_region = raw.get(
        "visualization",
        {},
    ).get(
        "previews_per_region",
        5,
    )

    if type(previews_per_region) is not int or previews_per_region < 0:
        raise ValueError("Invalid previews_per_region.")

    return variants, previews_per_region


def save_preview(
    preview_rgb: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
) -> None:
    """Save original image beside the agreement overlay."""

    comparison = np.concatenate(
        [preview_rgb, overlay],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(comparison).save(output_path)


def summarize_results(
    dataframe: pd.DataFrame,
    image_count: int,
    failure_counts: Counter,
) -> dict:
    """Summarize experiment results by variant and region."""

    summary = {
        "total_images": image_count,
        "variants": {},
        "processing_failures": dict(failure_counts),
        "clinical_validation": False,
        "segmentation_ground_truth_available": False,
        "metrics_measure_algorithm_agreement": True,
    }

    for variant in (
        "baseline",
        "moderate",
        "restrictive",
    ):
        subset = dataframe[dataframe["variant"] == variant]

        valid_iou = subset["iou"].dropna()
        valid_dice = subset["dice"].dropna()

        variant_summary = {
            "image_count": int(len(subset)),
            "mean_ycrcb_coverage": (
                float(subset["ycrcb_coverage"].mean()) if not subset.empty else None
            ),
            "mean_hsv_coverage": (
                float(subset["hsv_coverage"].mean()) if not subset.empty else None
            ),
            "mean_iou_agreement": (
                float(valid_iou.mean()) if not valid_iou.empty else None
            ),
            "mean_dice_agreement": (
                float(valid_dice.mean()) if not valid_dice.empty else None
            ),
            "mean_disagreement_rate": (
                float(subset["disagreement_rate"].mean()) if not subset.empty else None
            ),
            "both_masks_empty_count": int(subset["both_masks_empty"].sum()),
            "ycrcb_empty_count": int((subset["ycrcb_selected_pixels"] == 0).sum()),
            "hsv_empty_count": int((subset["hsv_selected_pixels"] == 0).sum()),
            "region_summary": {},
        }

        for region in ("forehead", "sternum"):
            region_subset = subset[subset["body_region"] == region]

            region_iou = region_subset["iou"].dropna()

            variant_summary["region_summary"][region] = {
                "image_count": int(len(region_subset)),
                "mean_ycrcb_coverage": (
                    float(region_subset["ycrcb_coverage"].mean())
                    if not region_subset.empty
                    else None
                ),
                "mean_hsv_coverage": (
                    float(region_subset["hsv_coverage"].mean())
                    if not region_subset.empty
                    else None
                ),
                "mean_iou_agreement": (
                    float(region_iou.mean()) if not region_iou.empty else None
                ),
            }

        summary["variants"][variant] = variant_summary

    return summary


def main() -> int:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.is_dir():
        print("[ERROR] Input directory does not exist.")
        return 2

    if input_root == output_root or input_root in output_root.parents:
        print("[ERROR] Output cannot be inside the raw dataset.")
        return 2

    try:
        preprocessing_config = load_preprocessing_config(args.preprocessing_config)

        roi_config = load_roi_config(args.roi_config)

        variants, previews_per_region = load_experiment(args.experiment_config)

        preprocessor = ImagePreprocessor(preprocessing_config)

        roi_extractor = GeometricROIExtractor(roi_config)

        segmenters = {name: SkinSegmenter(config) for name, config in variants.items()}

        image_paths = sorted(discover_images(input_root))

        if not image_paths:
            raise ValueError("No supported images found.")

        records = []
        failure_counts = Counter()

        preview_counts = {
            (variant, region): 0
            for variant in variants
            for region in ("forehead", "sternum")
        }

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

                for variant, segmenter in segmenters.items():
                    ycrcb = segmenter.segment(
                        preprocessed.preview_rgb,
                        roi.roi_mask,
                        "ycrcb",
                    )

                    hsv = segmenter.segment(
                        preprocessed.preview_rgb,
                        roi.roi_mask,
                        "hsv",
                    )

                    for result in (ycrcb, hsv):
                        qc = validate_segmentation_result(
                            result,
                            roi.roi_mask,
                        )

                        if not qc.passed:
                            raise ValueError(
                                "Segmentation QC failed: " + ",".join(qc.errors)
                            )

                    agreement = calculate_mask_agreement(
                        ycrcb.mask,
                        hsv.mask,
                        roi.roi_mask,
                    )

                    records.append(
                        {
                            "relative_path": relative_path,
                            "body_region": region,
                            "variant": variant,
                            "ycrcb_coverage": ycrcb.coverage,
                            "hsv_coverage": hsv.coverage,
                            "ycrcb_selected_pixels": (ycrcb.selected_pixel_count),
                            "hsv_selected_pixels": (hsv.selected_pixel_count),
                            "intersection_pixels": (agreement.intersection_pixels),
                            "union_pixels": (agreement.union_pixels),
                            "ycrcb_only_pixels": (agreement.ycrcb_only_pixels),
                            "hsv_only_pixels": (agreement.hsv_only_pixels),
                            "both_rejected_pixels": (agreement.both_rejected_pixels),
                            "geometric_roi_pixels": (agreement.geometric_roi_pixels),
                            "iou": agreement.iou,
                            "dice": agreement.dice,
                            "disagreement_rate": (agreement.disagreement_rate),
                            "both_masks_empty": (agreement.both_masks_empty),
                        }
                    )

                    key = (variant, region)

                    if preview_counts[key] < previews_per_region:
                        overlay = create_agreement_overlay(
                            preprocessed.preview_rgb,
                            ycrcb.mask,
                            hsv.mask,
                            roi.roi_mask,
                        )

                        preview_number = preview_counts[key] + 1

                        preview_path = (
                            output_root
                            / "previews"
                            / region
                            / variant
                            / (f"agreement_{preview_number:03d}.png")
                        )

                        save_preview(
                            preprocessed.preview_rgb,
                            overlay,
                            preview_path,
                        )

                        preview_counts[key] += 1

            except (ValueError, OSError) as exc:
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
            raise ValueError("No successful experiment results.")

        expected_rows = len(image_paths) * len(variants)

        if len(dataframe) != expected_rows:
            raise ValueError(
                "Incomplete experiment results: "
                f"expected {expected_rows}, "
                f"received {len(dataframe)}."
            )

        manifest_path = output_root / "agreement_manifest.csv"

        dataframe.to_csv(
            manifest_path,
            index=False,
        )

        summary = summarize_results(
            dataframe,
            len(image_paths),
            failure_counts,
        )

        summary["total_evaluations"] = len(dataframe)

        summary["status"] = "passed" if not failure_counts else "failed"

        summary_path = output_root / "agreement_summary.json"

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
        TypeError,
        yaml.YAMLError,
    ) as exc:
        LOGGER.error(
            "Experiment failed: %s",
            exc,
        )
        return 2

    print("=" * 60)
    print("NeoJaundice — Phase 5.5")
    print("Pixel-Level Agreement and Threshold Sensitivity")
    print("=" * 60)

    print(f"Images: {summary['total_images']}")

    print(f"Variants: {len(variants)}")

    print(f"Evaluations: {summary['total_evaluations']}")

    print(f"Status: {summary['status']}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

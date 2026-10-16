"""Prepare NeoJaundice manifests and patient-level splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from ml.datasets.neojaundice_adapter import (
    create_patient_splits,
    prepare_manifest,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NeoJaundice supervised datasets."
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
        default=Path("configs/datasets/neojaundice.yaml"),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        input_root = args.input.resolve()
        output_root = args.output.resolve()

        if not input_root.is_dir():
            raise FileNotFoundError(f"Dataset not found: {input_root}")

        if input_root == output_root or input_root in output_root.parents:
            raise ValueError("Output directory cannot be inside source dataset.")

        with args.config.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("Invalid YAML configuration.")

        if config.get("schema_version") != "1.0":
            raise ValueError("Unsupported configuration schema.")

        splitting = config["splitting"]

        manifest = prepare_manifest(input_root)

        splits = create_patient_splits(
            manifest,
            random_seed=int(splitting["random_seed"]),
            train_fraction=float(splitting["train_fraction"]),
            validation_fraction=float(splitting["validation_fraction"]),
            test_fraction=float(splitting["test_fraction"]),
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest.to_csv(
            output_root / "manifest.csv",
            index=False,
        )

        for name, dataframe in splits.items():
            dataframe.to_csv(
                output_root / f"{name}.csv",
                index=False,
            )

        summary = {
            "dataset": "NeoJaundice",
            "total_images": int(len(manifest)),
            "total_patients": int(manifest["patient_id"].nunique()),
            "zero_bilirubin_records": int(manifest["zero_bilirubin_review"].sum()),
            "missing_age_records": int(manifest["missing_age_review"].sum()),
            "splits": {},
            "patient_independent": True,
            "clinical_validation": False,
        }

        for name, dataframe in splits.items():
            summary["splits"][name] = {
                "images": int(len(dataframe)),
                "patients": int(dataframe["patient_id"].nunique()),
                "treatment_distribution": {
                    str(key): int(value)
                    for key, value in dataframe["treatment_label"]
                    .value_counts()
                    .items()
                },
                "mean_bilirubin": float(dataframe["bilirubin_value"].mean()),
            }

        all_patients = [set(df["patient_id"]) for df in splits.values()]

        if (
            all_patients[0] & all_patients[1]
            or all_patients[0] & all_patients[2]
            or all_patients[1] & all_patients[2]
        ):
            raise AssertionError("Patient leakage detected.")

        if sum(len(df) for df in splits.values()) != len(manifest):
            raise AssertionError("Split image counts do not match the manifest.")

        summary["status"] = "passed"

        summary_path = output_root / "dataset_summary.json"

        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("=" * 60)
        print("NeoJaundice — Phase 7.1")
        print("=" * 60)

        print(f"Total images: {summary['total_images']}")

        print(f"Total patients: {summary['total_patients']}")

        for name, details in summary["splits"].items():
            print(f"{name}: {details['images']} images, {details['patients']} patients")

        print(
            "Zero bilirubin records:",
            summary["zero_bilirubin_records"],
        )

        print(
            "Missing age records:",
            summary["missing_age_records"],
        )

        print("Patient leakage: NONE")
        print("Status: passed")

        return 0

    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        yaml.YAMLError,
        pd.errors.ParserError,
    ) as exc:
        LOGGER.error(
            "Dataset preparation failed: %s",
            exc,
        )
        return 2


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())

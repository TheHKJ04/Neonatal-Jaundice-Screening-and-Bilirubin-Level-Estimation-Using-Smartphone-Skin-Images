"""Command-line validation for candidate labeled datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.datasets.label_validator import (
    LabeledDatasetValidator,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate labeled neonatal image metadata."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        return 2

    if not args.image_root.is_dir():
        print(f"[ERROR] Image root not found: {args.image_root}")
        return 2

    try:
        dataframe = pd.read_csv(
            args.manifest,
            encoding="utf-8-sig",
        )

        validator = LabeledDatasetValidator(args.image_root)

        report = validator.validate(dataframe)

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                report.to_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )

    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        print(f"[ERROR] Validation failed: {exc}")
        return 2

    print("=" * 55)
    print("NeoJaundice — Labeled Dataset Validation")
    print("=" * 55)

    print(f"Total records: {report.total_records}")

    print(f"Valid records: {report.valid_records}")

    print(f"Invalid records: {report.invalid_records}")

    print(f"Status: {'PASSED' if report.passed else 'FAILED'}")

    print(f"Report: {args.output}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

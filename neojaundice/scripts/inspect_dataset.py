"""
Command-line dataset image inspector.

Example:
    python -m scripts.inspect_dataset \
        --input data/raw/example_dataset \
        --output data/interim/example_inspection
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.datasets.image_inspector import (
    inspect_dataset,
    summarize_inspection,
    write_manifest,
    write_summary,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Inspect image files and generate a technical dataset manifest.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dataset directory containing images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for inspection reports.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the dataset inspection workflow."""

    args = parse_arguments()

    dataset_root = args.input.resolve()

    output_root = args.output.resolve()

    if not dataset_root.is_dir():
        print("[FAILED] Dataset directory does not exist.")
        return 1

    # Prevent generated reports from being discovered
    # inside the input dataset directory.
    if output_root == dataset_root:
        print("[FAILED] Output cannot equal input.")
        return 1

    if dataset_root in output_root.parents:
        print("[FAILED] Output cannot be inside input.")
        return 1

    print("=" * 60)
    print("NeoJaundice Dataset Image Inspection")
    print("=" * 60)

    records = inspect_dataset(dataset_root)

    summary = summarize_inspection(records)

    manifest_path = output_root / "image_manifest.csv"

    summary_path = output_root / "inspection_summary.json"

    write_manifest(
        records,
        manifest_path,
    )

    write_summary(
        summary,
        summary_path,
    )

    print()
    print(f"Total images: {summary.total_images}")

    print(f"Valid images: {summary.valid_images}")

    print(f"Invalid images: {summary.invalid_images}")

    print(f"Exact duplicates: {summary.duplicate_images}")

    print(f"Unique valid images: {summary.unique_valid_images}")

    print()
    print("Reports generated:")
    print(manifest_path)
    print(summary_path)

    print()
    print("[PASSED] Dataset inspection completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

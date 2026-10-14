"""
CLI for NeoJaundice dataset structure analysis.

Example:
    python -m scripts.analyze_dataset_structure \
        --input data/raw/synthetic_demo \
        --output data/interim/synthetic_structure_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.datasets.structure_analyzer import (
    analyze_dataset_structure,
    write_structure_json,
    write_structure_markdown,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dataset structure and metadata headers."
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

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    dataset_root = args.input.resolve()
    output_root = args.output.resolve()

    if not dataset_root.is_dir():
        print("[FAILED] Dataset directory does not exist.")
        return 1

    if output_root == dataset_root or dataset_root in output_root.parents:
        print("[FAILED] Output cannot be inside the dataset.")
        return 1

    report = analyze_dataset_structure(dataset_root)

    if report.total_files == 0:
        print(
            "[FAILED] Dataset directory is empty. "
            "No files are available for inspection."
        )
        return 2

    json_path = output_root / "structure_report.json"
    markdown_path = output_root / "structure_report.md"

    write_structure_json(
        report,
        json_path,
    )

    write_structure_markdown(
        report,
        markdown_path,
    )

    print("=" * 60)
    print("NeoJaundice Dataset Structure Analysis")
    print("=" * 60)

    print(f"Dataset: {report.dataset_name}")
    print(f"Total files: {report.total_files}")
    print(f"Images: {report.image_files}")
    print(f"CSV files: {report.csv_files}")
    print(f"Excel files: {report.excel_files}")

    print()
    print(
        "Bilirubin column candidate:",
        report.has_bilirubin_column_candidate,
    )

    print(
        "Patient column candidate:",
        report.has_patient_column_candidate,
    )

    print(
        "Image column candidate:",
        report.has_image_column_candidate,
    )

    print()
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")

    print()
    print("[PASSED] Dataset structure analysis completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

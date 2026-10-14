"""
Dataset structure analysis for NeoJaundice.

This module inventories dataset files and inspects CSV
headers without loading complete clinical records.

It does not infer clinical validity from column names.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

CSV_EXTENSIONS = {
    ".csv",
}

EXCEL_EXTENSIONS = {
    ".xlsx",
    ".xls",
}

BILIRUBIN_TERMS = (
    "bilirubin",
    "tsb",
    "tcb",
)

PATIENT_TERMS = (
    "patient",
    "infant",
    "subject",
    "baby",
    "newborn",
)

IMAGE_TERMS = (
    "image",
    "photo",
    "filename",
    "file_name",
)

MAX_CSV_HEADER_BYTES = 1024 * 1024


@dataclass(frozen=True)
class MetadataFile:
    """Metadata discovered in a dataset."""

    relative_path: str
    file_type: str
    columns: list[str]
    candidate_bilirubin_columns: list[str]
    candidate_patient_columns: list[str]
    candidate_image_columns: list[str]
    inspection_error: str | None


@dataclass(frozen=True)
class StructureReport:
    """Technical dataset structure report."""

    dataset_name: str
    total_files: int
    image_files: int
    csv_files: int
    excel_files: int
    other_files: int
    image_extensions: dict[str, int]
    metadata_files: list[MetadataFile]
    has_bilirubin_column_candidate: bool
    has_patient_column_candidate: bool
    has_image_column_candidate: bool


def classify_column(
    column_name: str,
    terms: tuple[str, ...],
) -> bool:
    """Identify possible metadata fields by column name."""

    normalized = column_name.lower().strip()

    return any(term in normalized for term in terms)


def inspect_csv_header(
    csv_path: Path,
    dataset_root: Path,
) -> MetadataFile:
    """Inspect a CSV header without reading patient records."""

    relative_path = csv_path.relative_to(dataset_root).as_posix()

    try:
        with csv_path.open(
            "rb",
        ) as file:
            header_bytes = file.readline(MAX_CSV_HEADER_BYTES + 1)

        if len(header_bytes) > MAX_CSV_HEADER_BYTES:
            raise ValueError("CSV header exceeds configured size limit.")

        header_text = header_bytes.decode("utf-8-sig")

        columns = next(
            csv.reader([header_text]),
            [],
        )

        if not columns:
            raise ValueError("CSV header is empty.")

        return MetadataFile(
            relative_path=relative_path,
            file_type="csv",
            columns=columns,
            candidate_bilirubin_columns=[
                column
                for column in columns
                if classify_column(
                    column,
                    BILIRUBIN_TERMS,
                )
            ],
            candidate_patient_columns=[
                column
                for column in columns
                if classify_column(
                    column,
                    PATIENT_TERMS,
                )
            ],
            candidate_image_columns=[
                column
                for column in columns
                if classify_column(
                    column,
                    IMAGE_TERMS,
                )
            ],
            inspection_error=None,
        )

    except (
        OSError,
        UnicodeError,
        ValueError,
        csv.Error,
    ):
        return MetadataFile(
            relative_path=relative_path,
            file_type="csv",
            columns=[],
            candidate_bilirubin_columns=[],
            candidate_patient_columns=[],
            candidate_image_columns=[],
            inspection_error="csv_header_inspection_failed",
        )


def analyze_dataset_structure(
    dataset_root: Path,
) -> StructureReport:
    """Analyze dataset files without modifying them."""

    if dataset_root.is_symlink():
        raise ValueError("Dataset root cannot be a symbolic link.")

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_root}")

    image_files = 0
    csv_files = 0
    excel_files = 0
    other_files = 0

    extension_counts: Counter[str] = Counter()

    metadata_files: list[MetadataFile] = []

    for current_root, directories, filenames in os.walk(
        dataset_root,
        followlinks=False,
    ):
        current_directory = Path(current_root)

        directories[:] = [
            directory
            for directory in directories
            if not (current_directory / directory).is_symlink()
        ]

        for filename in sorted(filenames):
            file_path = current_directory / filename

            if file_path.is_symlink():
                continue

            extension = file_path.suffix.lower()

            if extension in IMAGE_EXTENSIONS:
                image_files += 1
                extension_counts[extension] += 1

            elif extension in CSV_EXTENSIONS:
                csv_files += 1

                metadata_files.append(
                    inspect_csv_header(
                        file_path,
                        dataset_root,
                    )
                )

            elif extension in EXCEL_EXTENSIONS:
                excel_files += 1

                metadata_files.append(
                    MetadataFile(
                        relative_path=(file_path.relative_to(dataset_root).as_posix()),
                        file_type="excel",
                        columns=[],
                        candidate_bilirubin_columns=[],
                        candidate_patient_columns=[],
                        candidate_image_columns=[],
                        inspection_error="excel_parsing_not_implemented",
                    )
                )

            else:
                other_files += 1

    metadata_files.sort(key=lambda item: item.relative_path)

    total_files = image_files + csv_files + excel_files + other_files

    return StructureReport(
        dataset_name=dataset_root.name,
        total_files=total_files,
        image_files=image_files,
        csv_files=csv_files,
        excel_files=excel_files,
        other_files=other_files,
        image_extensions=dict(sorted(extension_counts.items())),
        metadata_files=metadata_files,
        has_bilirubin_column_candidate=any(
            item.candidate_bilirubin_columns for item in metadata_files
        ),
        has_patient_column_candidate=any(
            item.candidate_patient_columns for item in metadata_files
        ),
        has_image_column_candidate=any(
            item.candidate_image_columns for item in metadata_files
        ),
    )


def write_structure_json(
    report: StructureReport,
    output_path: Path,
) -> None:
    """Write a structured JSON report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
        ),
        encoding="utf-8",
    )


def write_structure_markdown(
    report: StructureReport,
    output_path: Path,
) -> None:
    """Write a human-readable Markdown report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Dataset Structure Analysis",
        "",
        f"Dataset: {report.dataset_name}",
        "",
        "## File Inventory",
        "",
        f"- Total files: {report.total_files}",
        f"- Image files: {report.image_files}",
        f"- CSV files: {report.csv_files}",
        f"- Excel files: {report.excel_files}",
        f"- Other files: {report.other_files}",
        "",
        "## Candidate Clinical Metadata",
        "",
        (f"- Bilirubin column candidate: {report.has_bilirubin_column_candidate}"),
        (f"- Patient column candidate: {report.has_patient_column_candidate}"),
        (f"- Image column candidate: {report.has_image_column_candidate}"),
        "",
        "## Metadata Files",
        "",
    ]

    for item in report.metadata_files:
        lines.extend(
            [
                f"### `{item.relative_path}`",
                "",
                f"- Type: {item.file_type}",
                f"- Columns: {', '.join(item.columns) or 'Not inspected'}",
                (
                    "- Bilirubin candidates: "
                    f"{', '.join(item.candidate_bilirubin_columns) or 'None'}"
                ),
                (
                    "- Patient candidates: "
                    f"{', '.join(item.candidate_patient_columns) or 'None'}"
                ),
                (
                    "- Image candidates: "
                    f"{', '.join(item.candidate_image_columns) or 'None'}"
                ),
                f"- Inspection error: {item.inspection_error or 'None'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Research Interpretation",
            "",
            "Candidate column names do not establish clinical validity.",
            "",
            "Before regression training, verify:",
            "",
            "1. Numerical TSB reference measurements.",
            "2. Bilirubin units.",
            "3. Patient identifiers.",
            "4. Image-to-reference pairing.",
            "5. Measurement timestamps.",
            "6. Dataset permissions and provenance.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

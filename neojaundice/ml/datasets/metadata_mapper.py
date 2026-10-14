# type: ignore
"""
Clinical metadata integration for NeoJaundice.

Combines validated clinical metadata with technical
image inspection records.

This module does not determine clinical eligibility
or perform dataset splitting.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ml.datasets.schemas import DatasetSample

REQUIRED_COLUMNS = {
    "sample_id",
    "patient_id",
    "image_path",
    "captured_at",
    "bilirubin_measured_at",
    "bilirubin_type",
    "bilirubin_mg_dl",
    "age_hours",
    "gestational_age_weeks",
    "body_region",
}

OUTPUT_COLUMNS = (
    "sample_id",
    "patient_id",
    "image_path",
    "captured_at",
    "bilirubin_measured_at",
    "bilirubin_type",
    "bilirubin_mg_dl",
    "age_hours",
    "gestational_age_weeks",
    "body_region",
    "pairing_interval_minutes",
    "image_sha256",
    "image_width",
    "image_height",
    "image_format",
)


@dataclass(frozen=True)
class IntegrationResult:
    """Results from clinical metadata integration."""

    records: list[dict]
    errors: list[str]
    total_rows: int
    accepted_rows: int
    rejected_rows: int


def load_csv_records(
    csv_path: Path,
) -> list[dict[str, str]]:
    """Load CSV records with basic structural validation."""

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no header.")

        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV contains duplicate column names.")

        records = list(reader)

    return records


def validate_required_columns(
    csv_path: Path,
) -> None:
    """Verify clinical metadata has required columns."""

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        columns = set(reader.fieldnames or [])

    missing = REQUIRED_COLUMNS - columns

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))


def integrate_metadata(
    clinical_csv: Path,
    image_manifest_csv: Path,
) -> IntegrationResult:
    """Integrate clinical metadata with inspected images."""

    validate_required_columns(clinical_csv)

    clinical_records = load_csv_records(clinical_csv)

    image_records = load_csv_records(image_manifest_csv)

    image_index = {
        record["relative_path"]: record
        for record in image_records
        if record.get("status") == "valid"
        and record.get("duplicate_of")
        in (
            "",
            None,
        )
    }

    accepted: list[dict] = []
    errors: list[str] = []

    seen_sample_ids: set[str] = set()

    for row_number, row in enumerate(
        clinical_records,
        start=2,
    ):
        try:
            sample = DatasetSample.model_validate(row)

            if sample.sample_id in seen_sample_ids:
                raise ValueError("Duplicate sample_id.")

            seen_sample_ids.add(sample.sample_id)

            image_record = image_index.get(sample.image_path)

            if image_record is None:
                raise ValueError("Image not found in valid image manifest.")

            accepted.append(
                {
                    "sample_id": sample.sample_id,
                    "patient_id": sample.patient_id,
                    "image_path": sample.image_path,
                    "captured_at": (sample.captured_at.isoformat()),
                    "bilirubin_measured_at": (sample.bilirubin_measured_at.isoformat()),
                    "bilirubin_type": (sample.bilirubin_type.value),
                    "bilirubin_mg_dl": (sample.bilirubin_mg_dl),
                    "age_hours": sample.age_hours,
                    "gestational_age_weeks": (sample.gestational_age_weeks),
                    "body_region": (sample.body_region.value),
                    "pairing_interval_minutes": (sample.pairing_interval_minutes),
                    "image_sha256": image_record["sha256"],
                    "image_width": image_record["width"],
                    "image_height": image_record["height"],
                    "image_format": image_record["image_format"],
                }
            )

        except (
            ValidationError,
            ValueError,
            KeyError,
        ) as exc:
            errors.append(f"Row {row_number}: {exc}")

    return IntegrationResult(
        records=accepted,
        errors=errors,
        total_rows=len(clinical_records),
        accepted_rows=len(accepted),
        rejected_rows=len(errors),
    )


def write_standardized_manifest(
    records: list[dict],
    output_path: Path,
) -> None:
    """Write validated records to a standardized CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()

        writer.writerows(records)

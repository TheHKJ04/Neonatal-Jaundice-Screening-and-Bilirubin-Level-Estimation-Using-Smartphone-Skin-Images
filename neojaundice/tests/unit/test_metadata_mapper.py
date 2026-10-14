# type: ignore
"""
Tests for clinical metadata integration.
"""

import csv
from pathlib import Path

import pytest

from ml.datasets.metadata_mapper import (
    integrate_metadata,
    validate_required_columns,
    write_standardized_manifest,
)


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write synthetic CSV records."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def make_clinical_record(**overrides):
    """Create synthetic clinical metadata."""

    record = {
        "sample_id": "S001",
        "patient_id": "P001",
        "image_path": "sample.png",
        "captured_at": "2026-01-10T10:00:00+00:00",
        "bilirubin_measured_at": ("2026-01-10T10:30:00+00:00"),
        "bilirubin_type": "TSB",
        "bilirubin_mg_dl": "12.8",
        "age_hours": "72",
        "gestational_age_weeks": "38",
        "body_region": "forehead",
    }

    record.update(overrides)

    return record


def make_image_record(**overrides):
    """Create synthetic image manifest metadata."""

    record = {
        "relative_path": "sample.png",
        "status": "valid",
        "image_format": "PNG",
        "width": "128",
        "height": "128",
        "color_mode": "RGB",
        "file_size_bytes": "500",
        "sha256": "a" * 64,
        "duplicate_of": "",
        "error_code": "",
    }

    record.update(overrides)

    return record


def test_valid_metadata_integration(tmp_path: Path):
    clinical = tmp_path / "clinical.csv"
    images = tmp_path / "images.csv"

    write_csv(
        clinical,
        [make_clinical_record()],
    )

    write_csv(
        images,
        [make_image_record()],
    )

    result = integrate_metadata(
        clinical,
        images,
    )

    assert result.accepted_rows == 1
    assert result.rejected_rows == 0


def test_missing_image_rejected(tmp_path: Path):
    clinical = tmp_path / "clinical.csv"
    images = tmp_path / "images.csv"

    write_csv(
        clinical,
        [make_clinical_record(image_path="missing.png")],
    )

    write_csv(
        images,
        [make_image_record()],
    )

    result = integrate_metadata(
        clinical,
        images,
    )

    assert result.rejected_rows == 1


def test_negative_bilirubin_rejected(tmp_path: Path):
    clinical = tmp_path / "clinical.csv"
    images = tmp_path / "images.csv"

    write_csv(
        clinical,
        [make_clinical_record(bilirubin_mg_dl="-5")],
    )

    write_csv(
        images,
        [make_image_record()],
    )

    result = integrate_metadata(
        clinical,
        images,
    )

    assert result.rejected_rows == 1


def test_tcb_record_preserved(tmp_path: Path):
    clinical = tmp_path / "clinical.csv"
    images = tmp_path / "images.csv"

    write_csv(
        clinical,
        [make_clinical_record(bilirubin_type="TcB")],
    )

    write_csv(
        images,
        [make_image_record()],
    )

    result = integrate_metadata(
        clinical,
        images,
    )

    assert result.accepted_rows == 1

    assert result.records[0]["bilirubin_type"] == "TcB"


def test_missing_columns_rejected(tmp_path: Path):
    clinical = tmp_path / "clinical.csv"

    write_csv(
        clinical,
        [{"sample_id": "S001"}],
    )

    with pytest.raises(ValueError):
        validate_required_columns(clinical)


def test_manifest_written(tmp_path: Path):
    output = tmp_path / "standardized.csv"

    write_standardized_manifest(
        [
            {
                "sample_id": "S001",
                "patient_id": "P001",
                "image_path": "sample.png",
                "captured_at": "2026-01-10T10:00:00+00:00",
                "bilirubin_measured_at": ("2026-01-10T10:30:00+00:00"),
                "bilirubin_type": "TSB",
                "bilirubin_mg_dl": 12.8,
                "age_hours": 72.0,
                "gestational_age_weeks": 38.0,
                "body_region": "forehead",
                "pairing_interval_minutes": 30.0,
                "image_sha256": "a" * 64,
                "image_width": 128,
                "image_height": 128,
                "image_format": "PNG",
            }
        ],
        output,
    )

    assert output.exists()

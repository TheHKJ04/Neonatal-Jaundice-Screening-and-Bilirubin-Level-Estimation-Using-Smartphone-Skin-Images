"""Validate candidate datasets for supervised bilirubin regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "image_id",
    "patient_id",
    "image_path",
    "body_region",
    "bilirubin_value",
    "bilirubin_unit",
    "measurement_type",
}

ALLOWED_REGIONS = {"forehead", "sternum"}
ALLOWED_UNITS = {"mg/dL", "umol/L"}
ALLOWED_MEASUREMENT_TYPES = {"TSB", "TcB"}


@dataclass
class DatasetValidationReport:
    """Result of validating a candidate labeled dataset."""

    total_records: int
    valid_records: int = 0
    invalid_records: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": self.passed,
            "clinical_validation": False,
            "reference_measurements_verified": False,
        }


class LabeledDatasetValidator:
    """Perform structural checks on bilirubin regression metadata."""

    def __init__(self, image_root: Path) -> None:
        self.image_root = image_root.resolve()

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> DatasetValidationReport:
        report = DatasetValidationReport(total_records=len(dataframe))

        missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

        if missing_columns:
            report.errors.append(
                "Missing required columns: " + ", ".join(sorted(missing_columns))
            )
            report.invalid_records = len(dataframe)
            return report

        if dataframe.empty:
            report.errors.append("Dataset contains no records.")
            return report

        invalid_rows = set()

        for index, row in dataframe.iterrows():
            row_errors = []

            for column in REQUIRED_COLUMNS:
                value = row[column]

                if pd.isna(value) or not str(value).strip():
                    row_errors.append(f"Missing {column}")

            if row_errors:
                report.errors.append(f"Row {index}: " + "; ".join(row_errors))
                invalid_rows.add(index)
                continue

            try:
                bilirubin = float(row["bilirubin_value"])
            except (TypeError, ValueError):
                bilirubin = float("nan")

            if not np.isfinite(bilirubin) or bilirubin <= 0:
                row_errors.append("Invalid bilirubin value")

            if row["bilirubin_unit"] not in ALLOWED_UNITS:
                row_errors.append("Unsupported bilirubin unit")

            if row["measurement_type"] not in ALLOWED_MEASUREMENT_TYPES:
                row_errors.append("Unsupported measurement type")

            if row["body_region"] not in ALLOWED_REGIONS:
                row_errors.append("Unsupported anatomical region")

            relative_path = Path(str(row["image_path"]))

            if relative_path.is_absolute():
                row_errors.append("Absolute image paths are not allowed")
            else:
                image_path = (self.image_root / relative_path).resolve()

                if (
                    image_path != self.image_root
                    and self.image_root not in image_path.parents
                ):
                    row_errors.append("Image path escapes dataset root")
                elif not image_path.is_file():
                    row_errors.append("Image file does not exist")

            if row_errors:
                invalid_rows.add(index)

                report.errors.append(f"Row {index}: " + "; ".join(row_errors))

        duplicate_ids = dataframe["image_id"].duplicated(keep=False)

        for index in dataframe.index[duplicate_ids]:
            if index not in invalid_rows:
                report.errors.append(f"Row {index}: Duplicate image_id")
                invalid_rows.add(index)

        report.invalid_records = len(invalid_rows)

        report.valid_records = report.total_records - report.invalid_records

        report.passed = report.invalid_records == 0 and not report.errors

        if report.passed:
            report.warnings.append(
                "Structural validation passed. "
                "Reference measurements, dataset provenance, "
                "consent, and licensing still require verification."
            )

        return report

"""
Controlled dataset acquisition workflow for NeoJaundice.

The workflow:
- Checks dataset acquisition authorization
- Verifies archive integrity
- Inspects ZIP contents
- Extracts approved archives

It does not download datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.datasets.acquisition import (
    DatasetApproval,
    append_audit_event,
    extract_zip_safely,
    is_acquisition_authorized,
    verify_sha256,
)


@dataclass(frozen=True)
class AcquisitionResult:
    """Result of a controlled acquisition operation."""

    dataset_id: str
    destination: Path
    extracted_files: int
    verified_sha256: bool


def acquire_local_archive(
    *,
    approval: DatasetApproval,
    archive_path: Path,
    destination: Path,
    expected_sha256: str,
    audit_log: Path,
) -> AcquisitionResult:
    """
    Acquire an approved local ZIP archive.

    A trusted expected SHA-256 checksum is required.
    """

    dataset_id = approval.dataset_id

    if not is_acquisition_authorized(approval):
        append_audit_event(
            audit_log,
            dataset_id=dataset_id,
            action="acquisition",
            outcome="rejected_not_approved",
        )

        raise PermissionError(f"Dataset acquisition not approved: {dataset_id}")

    if not archive_path.is_file():
        append_audit_event(
            audit_log,
            dataset_id=dataset_id,
            action="acquisition",
            outcome="archive_missing",
        )

        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    if not verify_sha256(
        archive_path,
        expected_sha256,
    ):
        append_audit_event(
            audit_log,
            dataset_id=dataset_id,
            action="integrity_check",
            outcome="checksum_mismatch",
        )

        raise ValueError("Archive SHA-256 does not match the expected checksum.")

    append_audit_event(
        audit_log,
        dataset_id=dataset_id,
        action="integrity_check",
        outcome="passed",
    )

    extracted_paths = extract_zip_safely(
        archive_path,
        destination,
    )

    append_audit_event(
        audit_log,
        dataset_id=dataset_id,
        action="extraction",
        outcome="completed",
    )

    return AcquisitionResult(
        dataset_id=dataset_id,
        destination=destination,
        extracted_files=len(extracted_paths),
        verified_sha256=True,
    )

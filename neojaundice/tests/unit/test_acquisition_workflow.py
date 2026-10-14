"""
Unit tests for controlled dataset acquisition.
"""

import hashlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ml.datasets.acquisition import DatasetApproval
from ml.datasets.acquisition_workflow import acquire_local_archive


def make_approval(
    *,
    approved: bool,
) -> DatasetApproval:
    """Create synthetic acquisition approval metadata."""

    if not approved:
        return DatasetApproval.model_validate(
            {
                "dataset_id": "example_dataset",
                "status": "pending",
                "intended_use": "academic_research",
            }
        )

    return DatasetApproval.model_validate(
        {
            "dataset_id": "example_dataset",
            "status": "approved",
            "reviewed_by": "test_reviewer",
            "reviewed_at": datetime(
                2026,
                1,
                10,
                tzinfo=timezone.utc,
            ),
            "intended_use": "academic_research",
            "source_version": "1.0",
            "license_name": "Test License",
            "license_reviewed": True,
            "privacy_reviewed": True,
            "data_use_permission_confirmed": True,
            "approval_reference": "TEST-001",
        }
    )


def create_archive(
    path: Path,
    *,
    member_name: str = "images/sample.txt",
) -> str:
    """Create a synthetic ZIP archive."""

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            member_name,
            "synthetic test data",
        )

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pending_approval_rejected(tmp_path: Path):
    archive = tmp_path / "sample.zip"
    checksum = create_archive(archive)

    with pytest.raises(PermissionError):
        acquire_local_archive(
            approval=make_approval(approved=False),
            archive_path=archive,
            destination=tmp_path / "extracted",
            expected_sha256=checksum,
            audit_log=tmp_path / "audit.jsonl",
        )


def test_missing_archive_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        acquire_local_archive(
            approval=make_approval(approved=True),
            archive_path=tmp_path / "missing.zip",
            destination=tmp_path / "extracted",
            expected_sha256="a" * 64,
            audit_log=tmp_path / "audit.jsonl",
        )


def test_checksum_mismatch_rejected(tmp_path: Path):
    archive = tmp_path / "sample.zip"
    create_archive(archive)

    with pytest.raises(ValueError):
        acquire_local_archive(
            approval=make_approval(approved=True),
            archive_path=archive,
            destination=tmp_path / "extracted",
            expected_sha256="0" * 64,
            audit_log=tmp_path / "audit.jsonl",
        )


def test_approved_archive_extracted(tmp_path: Path):
    archive = tmp_path / "sample.zip"
    checksum = create_archive(archive)

    destination = tmp_path / "extracted"

    result = acquire_local_archive(
        approval=make_approval(approved=True),
        archive_path=archive,
        destination=destination,
        expected_sha256=checksum,
        audit_log=tmp_path / "audit.jsonl",
    )

    assert result.verified_sha256 is True
    assert result.extracted_files == 1

    assert (destination / "images" / "sample.txt").exists()


def test_unsafe_archive_rejected(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"

    checksum = create_archive(
        archive,
        member_name="../outside.txt",
    )

    with pytest.raises(ValueError):
        acquire_local_archive(
            approval=make_approval(approved=True),
            archive_path=archive,
            destination=tmp_path / "extracted",
            expected_sha256=checksum,
            audit_log=tmp_path / "audit.jsonl",
        )

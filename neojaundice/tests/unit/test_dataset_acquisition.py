"""
Unit tests for secure dataset acquisition utilities.
"""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from ml.datasets.acquisition import (
    DatasetApproval,
    append_audit_event,
    calculate_sha256,
    extract_zip_safely,
    inspect_zip_archive,
    is_acquisition_authorized,
    verify_sha256,
)


def make_approval(**overrides):
    """Create a synthetic acquisition approval record."""

    record = {
        "dataset_id": "example_dataset",
        "status": "pending",
        "reviewed_by": None,
        "reviewed_at": None,
        "intended_use": "academic_research",
        "source_version": None,
        "license_name": None,
        "license_url": None,
        "license_reviewed": False,
        "privacy_reviewed": False,
        "data_use_permission_confirmed": False,
        "redistribution_allowed": None,
        "approval_reference": None,
        "notes": "Synthetic test record.",
    }

    record.update(overrides)

    return record


def make_approved_record():
    """Create a complete synthetic approval record."""

    return make_approval(
        status="approved",
        reviewed_by="research_reviewer",
        reviewed_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
        source_version="1.0",
        license_name="Research License",
        license_reviewed=True,
        privacy_reviewed=True,
        data_use_permission_confirmed=True,
        approval_reference="TEST-APPROVAL-001",
    )


def test_pending_approval_is_valid():
    approval = DatasetApproval.model_validate(make_approval())

    assert approval.status.value == "pending"


def test_pending_dataset_not_authorized():
    approval = DatasetApproval.model_validate(make_approval())

    assert is_acquisition_authorized(approval) is False


def test_complete_approval_authorized():
    approval = DatasetApproval.model_validate(make_approved_record())

    assert is_acquisition_authorized(approval) is True


def test_incomplete_approval_rejected():
    record = make_approved_record()

    record["privacy_reviewed"] = False

    with pytest.raises(ValidationError):
        DatasetApproval.model_validate(record)


def test_missing_approval_reference_rejected():
    record = make_approved_record()

    record["approval_reference"] = None

    with pytest.raises(ValidationError):
        DatasetApproval.model_validate(record)


def test_sha256_calculation(tmp_path: Path):
    file_path = tmp_path / "example.bin"

    content = b"NeoJaundice test data"

    file_path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert calculate_sha256(file_path) == expected


def test_sha256_verification_success(tmp_path: Path):
    file_path = tmp_path / "example.bin"

    content = b"sample"

    file_path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert verify_sha256(
        file_path,
        expected,
    )


def test_sha256_verification_failure(tmp_path: Path):
    file_path = tmp_path / "example.bin"

    file_path.write_bytes(b"modified")

    expected = hashlib.sha256(b"original").hexdigest()

    assert not verify_sha256(
        file_path,
        expected,
    )


def test_invalid_sha256_rejected(tmp_path: Path):
    file_path = tmp_path / "example.bin"

    file_path.write_bytes(b"sample")

    with pytest.raises(ValueError):
        verify_sha256(
            file_path,
            "invalid-hash",
        )


def test_safe_zip_inspection(tmp_path: Path):
    archive_path = tmp_path / "safe.zip"

    with zipfile.ZipFile(
        archive_path,
        "w",
    ) as archive:
        archive.writestr(
            "images/sample.txt",
            "synthetic data",
        )

    members = inspect_zip_archive(archive_path)

    assert len(members) == 1


def test_zip_path_traversal_rejected(tmp_path: Path):
    archive_path = tmp_path / "unsafe.zip"

    with zipfile.ZipFile(
        archive_path,
        "w",
    ) as archive:
        archive.writestr(
            "../outside.txt",
            "unsafe",
        )

    with pytest.raises(ValueError):
        inspect_zip_archive(archive_path)


def test_safe_zip_extraction(tmp_path: Path):
    archive_path = tmp_path / "safe.zip"

    destination = tmp_path / "extracted"

    with zipfile.ZipFile(
        archive_path,
        "w",
    ) as archive:
        archive.writestr(
            "images/sample.txt",
            "synthetic data",
        )

    extracted = extract_zip_safely(
        archive_path,
        destination,
    )

    assert len(extracted) == 1

    assert (destination / "images" / "sample.txt").read_text(
        encoding="utf-8"
    ) == "synthetic data"


def test_existing_destination_rejected(tmp_path: Path):
    archive_path = tmp_path / "safe.zip"

    destination = tmp_path / "existing"

    destination.mkdir()

    with zipfile.ZipFile(
        archive_path,
        "w",
    ) as archive:
        archive.writestr(
            "sample.txt",
            "synthetic data",
        )

    with pytest.raises(FileExistsError):
        extract_zip_safely(
            archive_path,
            destination,
        )


def test_audit_event(tmp_path: Path):
    log_path = tmp_path / "audit.jsonl"

    append_audit_event(
        log_path,
        dataset_id="example_dataset",
        action="integrity_check",
        outcome="passed",
    )

    events = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(events) == 1

    assert events[0]["dataset_id"] == "example_dataset"

    assert events[0]["outcome"] == "passed"

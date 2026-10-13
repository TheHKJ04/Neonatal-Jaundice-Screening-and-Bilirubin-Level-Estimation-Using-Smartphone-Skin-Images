"""
Secure dataset acquisition utilities for NeoJaundice.

This module does not download datasets.

It provides approval validation, integrity checking,
safe ZIP extraction, and structured audit logging.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

HASH_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")

MAX_ARCHIVE_MEMBERS = 10_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_COMPRESSION_RATIO = 100


class ApprovalStatus(str, Enum):
    """Supported acquisition approval statuses."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUSPENDED = "suspended"


class DatasetApproval(BaseModel):
    """Approval metadata for one dataset."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    dataset_id: str = Field(
        min_length=3,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    status: ApprovalStatus

    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    intended_use: str = Field(min_length=3)

    source_version: str | None = None

    license_name: str | None = None
    license_url: str | None = None

    license_reviewed: bool = False
    privacy_reviewed: bool = False
    data_use_permission_confirmed: bool = False

    redistribution_allowed: bool | None = None

    approval_reference: str | None = None

    notes: str = ""

    @model_validator(mode="after")
    def validate_approval(self) -> DatasetApproval:
        """Require completed review before approval."""

        if self.status != ApprovalStatus.APPROVED:
            return self

        required_flags = (
            self.license_reviewed,
            self.privacy_reviewed,
            self.data_use_permission_confirmed,
        )

        if not all(required_flags):
            raise ValueError(
                "Approved datasets require completed "
                "license, privacy, and permission reviews."
            )

        required_metadata = (
            self.reviewed_by,
            self.reviewed_at,
            self.source_version,
            self.license_name,
            self.approval_reference,
        )

        if any(value is None for value in required_metadata):
            raise ValueError(
                "Approved datasets require reviewer, date, "
                "source version, license, and approval reference."
            )

        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Approval timestamp must include a timezone.")

        return self


def is_acquisition_authorized(
    approval: DatasetApproval,
) -> bool:
    """Return whether acquisition is approved."""

    return approval.status == ApprovalStatus.APPROVED


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256 without loading the whole file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def verify_sha256(
    file_path: Path,
    expected_hash: str,
) -> bool:
    """Compare file SHA-256 against a supplied reference."""

    if not HASH_PATTERN.fullmatch(expected_hash):
        raise ValueError("Expected SHA-256 must contain 64 hexadecimal characters.")

    actual_hash = calculate_sha256(file_path)

    return actual_hash.lower() == expected_hash.lower()


def _validate_zip_member(
    member: zipfile.ZipInfo,
) -> None:
    """Reject unsafe ZIP member names and file types."""

    name = member.filename

    if "\\" in name:
        raise ValueError("ZIP member contains a backslash.")

    path = PurePosixPath(name)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError("ZIP member contains an unsafe path.")

    if not name or name.startswith("./"):
        raise ValueError("ZIP member has an invalid name.")

    if ":" in name or "\x00" in name:
        raise ValueError("ZIP member contains invalid characters.")

    mode = (member.external_attr >> 16) & 0xFFFF

    file_type = stat.S_IFMT(mode)

    if file_type not in (
        0,
        stat.S_IFREG,
        stat.S_IFDIR,
    ):
        raise ValueError("ZIP member contains an unsupported file type.")

    if member.flag_bits & 0x1:
        raise ValueError("Encrypted ZIP members are not supported.")


def inspect_zip_archive(
    archive_path: Path,
) -> list[zipfile.ZipInfo]:
    """Inspect a ZIP archive before extraction."""

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()

        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("ZIP archive contains too many members.")

        total_size = 0
        seen_paths: set[str] = set()

        for member in members:
            _validate_zip_member(member)

            normalized = member.filename.rstrip("/")

            if normalized in seen_paths:
                raise ValueError("ZIP archive contains duplicate member paths.")

            seen_paths.add(normalized)

            total_size += member.file_size

            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("ZIP archive exceeds the uncompressed size limit.")

            if member.file_size > 0:
                ratio = member.file_size / max(member.compress_size, 1)

                if ratio > MAX_COMPRESSION_RATIO:
                    raise ValueError("ZIP member exceeds compression ratio limit.")

        return members


def extract_zip_safely(
    archive_path: Path,
    destination: Path,
) -> list[Path]:
    """
    Extract a verified ZIP archive to a new destination.

    The destination must not already exist.

    Extraction occurs in a temporary sibling directory.
    The completed directory is then moved into place.
    """

    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Extraction destination must not already exist.")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    members = inspect_zip_archive(archive_path)

    extracted_paths: list[Path] = []

    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=".dataset_extract_",
    ) as temporary_directory:
        staging = Path(temporary_directory)

        with zipfile.ZipFile(archive_path) as archive:
            for member in members:
                relative_path = PurePosixPath(member.filename)

                target = staging.joinpath(*relative_path.parts)

                if member.is_dir():
                    target.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    continue

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with archive.open(member) as source:
                    with target.open("xb") as output:
                        remaining = member.file_size

                        while remaining > 0:
                            chunk = source.read(min(1024 * 1024, remaining))

                            if not chunk:
                                raise ValueError("Unexpected end of ZIP member.")

                            output.write(chunk)
                            remaining -= len(chunk)

                        if source.read(1):
                            raise ValueError("ZIP member exceeds declared size.")

                extracted_paths.append(Path(*relative_path.parts))

        os.rename(staging, destination)

    return [destination / relative_path for relative_path in extracted_paths]


def append_audit_event(
    log_path: Path,
    *,
    dataset_id: str,
    action: str,
    outcome: str,
) -> None:
    """
    Append a minimal acquisition audit event.

    Do not include patient data, credentials,
    or complete local filesystem paths.
    """

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "action": action,
        "outcome": outcome,
    }

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(event) + "\n")

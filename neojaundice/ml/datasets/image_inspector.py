"""
Image inspection and manifest generation for NeoJaundice.

Responsibilities:
- Discover supported image files
- Validate image decoding
- Extract technical image metadata
- Calculate SHA-256 fingerprints
- Detect exact duplicates
- Generate CSV and JSON reports

This module does not:
- Modify original images
- Extract clinical labels
- Identify patients
- Perform image preprocessing
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

SUPPORTED_FORMATS = {
    "JPEG",
    "PNG",
}

MAX_IMAGE_PIXELS = 50_000_000

MAX_IMAGE_BYTES = 100 * 1024 * 1024

MANIFEST_FIELDS = (
    "relative_path",
    "status",
    "image_format",
    "width",
    "height",
    "color_mode",
    "file_size_bytes",
    "sha256",
    "duplicate_of",
    "error_code",
)


@dataclass(frozen=True)
class ImageRecord:
    """Technical metadata for one image file."""

    relative_path: str
    status: str
    image_format: str | None
    width: int | None
    height: int | None
    color_mode: str | None
    file_size_bytes: int
    sha256: str | None
    duplicate_of: str | None
    error_code: str | None


@dataclass(frozen=True)
class InspectionSummary:
    """Aggregated dataset inspection results."""

    total_images: int
    valid_images: int
    invalid_images: int
    duplicate_images: int
    unique_valid_images: int
    image_formats: dict[str, int]
    color_modes: dict[str, int]
    error_codes: dict[str, int]


def calculate_file_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate a file's SHA-256 fingerprint."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def discover_images(
    dataset_root: Path,
) -> list[Path]:
    """
    Discover supported image files recursively.

    Directory symbolic links are not followed.
    Symbolic-link files are retained for explicit rejection.
    """

    if dataset_root.is_symlink():
        raise ValueError("Dataset root cannot be a symbolic link")

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_root}")

    discovered: list[Path] = []

    for current_root, directory_names, file_names in os.walk(
        dataset_root,
        followlinks=False,
    ):
        current_directory = Path(current_root)

        directory_names[:] = [
            name
            for name in directory_names
            if not (current_directory / name).is_symlink()
        ]

        for file_name in file_names:
            path = current_directory / file_name

            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                discovered.append(path)

    return sorted(
        discovered,
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )


def _invalid_record(
    *,
    relative_path: str,
    file_size_bytes: int,
    error_code: str,
) -> ImageRecord:
    """Create a standardized invalid-image record."""

    return ImageRecord(
        relative_path=relative_path,
        status="invalid",
        image_format=None,
        width=None,
        height=None,
        color_mode=None,
        file_size_bytes=file_size_bytes,
        sha256=None,
        duplicate_of=None,
        error_code=error_code,
    )


def inspect_image(
    file_path: Path,
    dataset_root: Path,
) -> ImageRecord:
    """
    Inspect one image without modifying it.

    The caller must provide a file beneath dataset_root.
    """

    relative_path = file_path.relative_to(dataset_root).as_posix()

    if file_path.is_symlink():
        return _invalid_record(
            relative_path=relative_path,
            file_size_bytes=0,
            error_code="symbolic_link",
        )

    try:
        file_size = file_path.stat().st_size

        if file_size == 0:
            return _invalid_record(
                relative_path=relative_path,
                file_size_bytes=0,
                error_code="empty_file",
            )

        if file_size > MAX_IMAGE_BYTES:
            return _invalid_record(
                relative_path=relative_path,
                file_size_bytes=file_size,
                error_code="file_too_large",
            )

        with Image.open(file_path) as image:
            image_format = image.format
            width, height = image.size
            color_mode = image.mode

            if image_format not in SUPPORTED_FORMATS:
                return _invalid_record(
                    relative_path=relative_path,
                    file_size_bytes=file_size,
                    error_code="unsupported_format",
                )

            if width <= 0 or height <= 0:
                return _invalid_record(
                    relative_path=relative_path,
                    file_size_bytes=file_size,
                    error_code="invalid_dimensions",
                )

            if width * height > MAX_IMAGE_PIXELS:
                return _invalid_record(
                    relative_path=relative_path,
                    file_size_bytes=file_size,
                    error_code="pixel_limit_exceeded",
                )

            # Force full decoding to detect truncated/corrupt data.
            image.load()

        fingerprint = calculate_file_sha256(file_path)

        return ImageRecord(
            relative_path=relative_path,
            status="valid",
            image_format=image_format,
            width=width,
            height=height,
            color_mode=color_mode,
            file_size_bytes=file_size,
            sha256=fingerprint,
            duplicate_of=None,
            error_code=None,
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        return _invalid_record(
            relative_path=relative_path,
            file_size_bytes=(file_path.stat().st_size if file_path.exists() else 0),
            error_code="decode_error",
        )


def inspect_dataset(
    dataset_root: Path,
) -> list[ImageRecord]:
    """
    Inspect all supported images in a dataset.

    Exact duplicates are identified using SHA-256.
    """

    image_paths = discover_images(dataset_root)

    records: list[ImageRecord] = []

    first_seen_hashes: dict[str, str] = {}

    for file_path in image_paths:
        record = inspect_image(
            file_path=file_path,
            dataset_root=dataset_root,
        )

        if record.status == "valid" and record.sha256 is not None:
            original_path = first_seen_hashes.get(record.sha256)

            if original_path is not None:
                record = ImageRecord(
                    **{
                        **asdict(record),
                        "duplicate_of": original_path,
                    }
                )

            else:
                first_seen_hashes[record.sha256] = record.relative_path

        records.append(record)

    return records


def summarize_inspection(
    records: list[ImageRecord],
) -> InspectionSummary:
    """Calculate dataset inspection statistics."""

    valid_records = [record for record in records if record.status == "valid"]

    invalid_records = [record for record in records if record.status == "invalid"]

    duplicate_records = [
        record for record in valid_records if record.duplicate_of is not None
    ]

    formats = Counter(
        record.image_format
        for record in valid_records
        if record.image_format is not None
    )

    color_modes = Counter(
        record.color_mode for record in valid_records if record.color_mode is not None
    )

    errors = Counter(
        record.error_code for record in invalid_records if record.error_code is not None
    )

    return InspectionSummary(
        total_images=len(records),
        valid_images=len(valid_records),
        invalid_images=len(invalid_records),
        duplicate_images=len(duplicate_records),
        unique_valid_images=(len(valid_records) - len(duplicate_records)),
        image_formats=dict(formats),
        color_modes=dict(color_modes),
        error_codes=dict(errors),
    )


def write_manifest(
    records: list[ImageRecord],
    output_path: Path,
) -> None:
    """Write image inspection results to CSV."""

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
            fieldnames=MANIFEST_FIELDS,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def write_summary(
    summary: InspectionSummary,
    output_path: Path,
) -> None:
    """Write aggregated inspection results to JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            asdict(summary),
            indent=2,
        ),
        encoding="utf-8",
    )

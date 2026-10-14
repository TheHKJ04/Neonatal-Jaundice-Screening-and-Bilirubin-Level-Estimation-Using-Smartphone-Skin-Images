"""
Unit tests for NeoJaundice image inspection.
"""

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from ml.datasets.image_inspector import (
    calculate_file_sha256,
    discover_images,
    inspect_dataset,
    inspect_image,
    summarize_inspection,
    write_manifest,
    write_summary,
)


def create_image(
    path: Path,
    *,
    size: tuple[int, int] = (64, 64),
    color: tuple[int, int, int] = (255, 220, 180),
) -> None:
    """Create a synthetic RGB image."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.new(
        "RGB",
        size,
        color,
    )

    image.save(path)


def test_discover_images(tmp_path: Path):
    create_image(tmp_path / "a.jpg")

    create_image(tmp_path / "nested" / "b.png")

    (tmp_path / "notes.txt").write_text(
        "Not an image",
        encoding="utf-8",
    )

    images = discover_images(tmp_path)

    assert len(images) == 2


def test_missing_dataset_directory(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        discover_images(tmp_path / "missing")


def test_valid_jpeg_image(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"

    create_image(image_path)

    record = inspect_image(
        image_path,
        tmp_path,
    )

    assert record.status == "valid"
    assert record.image_format == "JPEG"
    assert record.width == 64
    assert record.height == 64
    assert record.color_mode == "RGB"


def test_valid_png_image(tmp_path: Path):
    image_path = tmp_path / "sample.png"

    create_image(image_path)

    record = inspect_image(
        image_path,
        tmp_path,
    )

    assert record.status == "valid"
    assert record.image_format == "PNG"


def test_corrupted_image(tmp_path: Path):
    image_path = tmp_path / "corrupted.jpg"

    image_path.write_bytes(b"not a valid image")

    record = inspect_image(
        image_path,
        tmp_path,
    )

    assert record.status == "invalid"
    assert record.error_code == "decode_error"


def test_empty_image(tmp_path: Path):
    image_path = tmp_path / "empty.jpg"

    image_path.write_bytes(b"")

    record = inspect_image(
        image_path,
        tmp_path,
    )

    assert record.status == "invalid"
    assert record.error_code == "empty_file"


def test_sha256_consistency(tmp_path: Path):
    image_path = tmp_path / "sample.png"

    create_image(image_path)

    first_hash = calculate_file_sha256(image_path)

    second_hash = calculate_file_sha256(image_path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_sha256_changes_when_file_changes(
    tmp_path: Path,
):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    create_image(
        first,
        color=(255, 0, 0),
    )

    create_image(
        second,
        color=(0, 255, 0),
    )

    assert calculate_file_sha256(first) != calculate_file_sha256(second)


def test_exact_duplicate_detection(
    tmp_path: Path,
):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"

    create_image(first)

    second.write_bytes(first.read_bytes())

    records = inspect_dataset(tmp_path)

    summary = summarize_inspection(records)

    assert summary.total_images == 2
    assert summary.duplicate_images == 1
    assert summary.unique_valid_images == 1


def test_different_images_not_duplicates(
    tmp_path: Path,
):
    create_image(
        tmp_path / "a.png",
        color=(255, 0, 0),
    )

    create_image(
        tmp_path / "b.png",
        color=(0, 0, 255),
    )

    records = inspect_dataset(tmp_path)

    summary = summarize_inspection(records)

    assert summary.duplicate_images == 0


def test_inspection_summary(tmp_path: Path):
    create_image(tmp_path / "valid.jpg")

    (tmp_path / "invalid.jpg").write_bytes(b"corrupted")

    records = inspect_dataset(tmp_path)

    summary = summarize_inspection(records)

    assert summary.total_images == 2
    assert summary.valid_images == 1
    assert summary.invalid_images == 1


def test_manifest_generation(tmp_path: Path):
    image_root = tmp_path / "images"

    image_root.mkdir()

    create_image(image_root / "sample.png")

    records = inspect_dataset(image_root)

    manifest_path = tmp_path / "output" / "manifest.csv"

    write_manifest(
        records,
        manifest_path,
    )

    assert manifest_path.exists()

    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["status"] == "valid"


def test_summary_generation(tmp_path: Path):
    create_image(tmp_path / "sample.png")

    records = inspect_dataset(tmp_path)

    summary = summarize_inspection(records)

    output_path = tmp_path / "reports" / "summary.json"

    write_summary(
        summary,
        output_path,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["total_images"] == 1
    assert report["valid_images"] == 1


def test_nested_relative_paths(tmp_path: Path):
    image_path = tmp_path / "forehead" / "sample.png"

    create_image(image_path)

    record = inspect_image(
        image_path,
        tmp_path,
    )

    assert record.relative_path == "forehead/sample.png"


def test_unsupported_extension_not_discovered(
    tmp_path: Path,
):
    (tmp_path / "example.txt").write_text(
        "example",
        encoding="utf-8",
    )

    images = discover_images(tmp_path)

    assert images == []

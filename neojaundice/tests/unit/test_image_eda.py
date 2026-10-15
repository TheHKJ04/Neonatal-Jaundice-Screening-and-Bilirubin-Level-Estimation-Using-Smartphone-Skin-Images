"""Unit tests for NeoJaundice image EDA."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.datasets.image_eda import (
    analyze_dataset,
    analyze_image,
    build_dataset_summary,
    discover_images,
)


def create_image(
    path: Path,
    value: int = 128,
    width: int = 64,
    height: int = 48,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = np.full(
        (height, width, 3),
        value,
        dtype=np.uint8,
    )

    assert cv2.imwrite(str(path), image)


def test_image_discovery(tmp_path: Path):
    create_image(tmp_path / "forehead" / "a.png")
    create_image(tmp_path / "sternum" / "b.jpg")

    images = discover_images(tmp_path)

    assert len(images) == 2


def test_image_dimensions(tmp_path: Path):
    image_path = tmp_path / "sample.png"

    create_image(
        image_path,
        width=80,
        height=40,
    )

    result = analyze_image(
        image_path,
        tmp_path,
    )

    assert result.width == 80
    assert result.height == 40
    assert result.aspect_ratio == pytest.approx(2.0)


def test_constant_image_brightness(tmp_path: Path):
    image_path = tmp_path / "sample.png"

    create_image(
        image_path,
        value=128,
    )

    result = analyze_image(
        image_path,
        tmp_path,
    )

    assert result.brightness == pytest.approx(128.0)


def test_constant_image_contrast(tmp_path: Path):
    image_path = tmp_path / "sample.png"

    create_image(
        image_path,
        value=128,
    )

    result = analyze_image(
        image_path,
        tmp_path,
    )

    assert result.contrast == pytest.approx(0.0)


def test_constant_image_blur_score(tmp_path: Path):
    image_path = tmp_path / "sample.png"

    create_image(
        image_path,
        value=128,
    )

    result = analyze_image(
        image_path,
        tmp_path,
    )

    assert result.blur_score == pytest.approx(0.0)


def test_white_image_overexposure(tmp_path: Path):
    image_path = tmp_path / "white.png"

    create_image(
        image_path,
        value=255,
    )

    result = analyze_image(
        image_path,
        tmp_path,
    )

    assert result.overexposed_fraction == pytest.approx(1.0)


def test_black_image_underexposure(tmp_path: Path):
    image_path = tmp_path / "black.png"

    create_image(
        image_path,
        value=0,
    )

    result = analyze_image(
        image_path,
        tmp_path,
    )

    assert result.underexposed_fraction == pytest.approx(1.0)


def test_region_folder_detection(tmp_path: Path):
    image_path = tmp_path / "forehead" / "sample.png"

    create_image(image_path)

    result = analyze_image(
        image_path,
        tmp_path,
    )

    assert result.body_region_folder == "forehead"


def test_dataset_analysis(tmp_path: Path):
    create_image(tmp_path / "a.png")
    create_image(tmp_path / "b.png")

    dataframe, failures = analyze_dataset(tmp_path)

    assert len(dataframe) == 2
    assert failures == []


def test_dataset_summary(tmp_path: Path):
    create_image(tmp_path / "a.png")

    dataframe, failures = analyze_dataset(tmp_path)

    summary = build_dataset_summary(
        dataframe,
        failures,
    )

    assert summary["analyzed_images"] == 1
    assert summary["failed_images"] == 0


def test_empty_dataset_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        analyze_dataset(tmp_path)


def test_invalid_image_recorded(tmp_path: Path):
    create_image(tmp_path / "valid.png")

    (tmp_path / "invalid.jpg").write_bytes(b"not a valid image")

    dataframe, failures = analyze_dataset(tmp_path)

    assert len(dataframe) == 1
    assert len(failures) == 1

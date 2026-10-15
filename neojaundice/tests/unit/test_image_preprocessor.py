"""Unit tests for NeoJaundice image preprocessing."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from ml.preprocessing.image_preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
    load_preprocessing_config,
)


def create_image(
    path: Path,
    width: int = 640,
    height: int = 480,
    color: tuple[int, int, int] = (200, 150, 100),
) -> None:
    image = Image.new(
        "RGB",
        (width, height),
        color=color,
    )

    image.save(path)


def test_default_configuration():
    config = PreprocessingConfig()

    assert config.target_width == 224
    assert config.target_height == 224


def test_invalid_width_rejected():
    with pytest.raises(ValueError):
        PreprocessingConfig(target_width=0)


def test_invalid_height_rejected():
    with pytest.raises(ValueError):
        PreprocessingConfig(target_height=-1)


def test_invalid_interpolation_rejected():
    with pytest.raises(ValueError):
        PreprocessingConfig(interpolation="invalid")


def test_invalid_normalization_rejected():
    with pytest.raises(ValueError):
        PreprocessingConfig(normalization="invalid")


def test_invalid_padding_rejected():
    with pytest.raises(ValueError):
        PreprocessingConfig(padding_color=(0, 0, 999))


def test_rgb_tensor_shape(tmp_path: Path):
    path = tmp_path / "sample.png"
    create_image(path)

    result = ImagePreprocessor(PreprocessingConfig()).process(path)

    assert result.image_tensor.shape == (
        3,
        224,
        224,
    )


def test_valid_mask_shape(tmp_path: Path):
    path = tmp_path / "sample.png"
    create_image(path)

    result = ImagePreprocessor(PreprocessingConfig()).process(path)

    assert result.valid_mask.shape == (
        1,
        224,
        224,
    )


def test_aspect_ratio_preserved(tmp_path: Path):
    path = tmp_path / "sample.png"
    create_image(path)

    result = ImagePreprocessor(PreprocessingConfig()).process(path)

    assert result.metadata.resized_width == 224
    assert result.metadata.resized_height == 168


def test_padding_geometry(tmp_path: Path):
    path = tmp_path / "sample.png"
    create_image(path)

    result = ImagePreprocessor(PreprocessingConfig()).process(path)

    assert result.metadata.padding_top == 28
    assert result.metadata.padding_bottom == 28
    assert result.metadata.padding_left == 0
    assert result.metadata.padding_right == 0


def test_mask_excludes_padding(tmp_path: Path):
    path = tmp_path / "sample.png"
    create_image(path)

    result = ImagePreprocessor(PreprocessingConfig()).process(path)

    assert result.valid_mask[:, :28, :].sum() == 0
    assert result.valid_mask[:, 28:196, :].all()
    assert result.valid_mask[:, 196:, :].sum() == 0


def test_tensor_dtype(tmp_path: Path):
    path = tmp_path / "sample.png"
    create_image(path)

    result = ImagePreprocessor(PreprocessingConfig()).process(path)

    assert result.image_tensor.dtype == torch.float32


def test_unit_normalization(tmp_path: Path):
    path = tmp_path / "sample.png"

    create_image(
        path,
        width=224,
        height=224,
        color=(255, 0, 0),
    )

    result = ImagePreprocessor(PreprocessingConfig(normalization="unit")).process(path)

    assert result.image_tensor[0, 0, 0] == pytest.approx(1.0)

    assert result.image_tensor[1, 0, 0] == pytest.approx(0.0)


def test_imagenet_normalization(tmp_path: Path):
    path = tmp_path / "sample.png"

    create_image(
        path,
        width=224,
        height=224,
        color=(255, 0, 0),
    )

    result = ImagePreprocessor(PreprocessingConfig(normalization="imagenet")).process(
        path
    )

    expected_red = (1.0 - 0.485) / 0.229

    assert result.image_tensor[0, 0, 0] == pytest.approx(
        expected_red,
        rel=1e-5,
    )


def test_missing_image_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ImagePreprocessor(PreprocessingConfig()).process(tmp_path / "missing.png")


def test_invalid_image_rejected(tmp_path: Path):
    path = tmp_path / "invalid.png"

    path.write_bytes(b"not an image")

    with pytest.raises(ValueError):
        ImagePreprocessor(PreprocessingConfig()).process(path)


def test_oversized_image_rejected(tmp_path: Path):
    path = tmp_path / "sample.png"

    create_image(
        path,
        width=100,
        height=100,
    )

    with pytest.raises(ValueError):
        ImagePreprocessor(PreprocessingConfig(max_input_pixels=100)).process(path)


def test_grayscale_converted_to_rgb(tmp_path: Path):
    path = tmp_path / "gray.png"

    Image.new(
        "L",
        (224, 224),
        color=128,
    ).save(path)

    result = ImagePreprocessor(PreprocessingConfig(normalization="unit")).process(path)

    assert result.image_tensor.shape[0] == 3

    assert torch.allclose(
        result.image_tensor[0],
        result.image_tensor[1],
    )


def test_exif_orientation_applied(tmp_path: Path):
    path = tmp_path / "oriented.jpg"

    image = Image.new(
        "RGB",
        (80, 40),
        color=(100, 150, 200),
    )

    exif = Image.Exif()
    exif[274] = 6

    image.save(
        path,
        exif=exif,
    )

    result = ImagePreprocessor(PreprocessingConfig()).process(path)

    assert result.metadata.original_width == 80
    assert result.metadata.original_height == 40

    assert result.metadata.oriented_width == 40
    assert result.metadata.oriented_height == 80


def test_load_yaml_config(tmp_path: Path):
    path = tmp_path / "config.yaml"

    config = {
        "schema_version": "1.0",
        "image": {
            "target_width": 128,
            "target_height": 128,
            "interpolation": "bilinear",
            "padding_color": [0, 0, 0],
        },
        "normalization": {
            "mode": "unit",
        },
        "safety": {
            "max_input_pixels": 1000000,
        },
    }

    path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    loaded = load_preprocessing_config(path)

    assert loaded.target_width == 128
    assert loaded.normalization == "unit"


def test_missing_yaml_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_preprocessing_config(tmp_path / "missing.yaml")


def test_preview_shape(tmp_path: Path):
    path = tmp_path / "sample.png"
    create_image(path)

    result = ImagePreprocessor(PreprocessingConfig()).process(path)

    assert result.preview_rgb.shape == (
        224,
        224,
        3,
    )

    assert result.preview_rgb.dtype == np.uint8

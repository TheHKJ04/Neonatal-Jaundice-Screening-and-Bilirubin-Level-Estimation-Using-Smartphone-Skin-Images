"""Configuration-driven image preprocessing for NeoJaundice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

InterpolationMode = Literal[
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos",
]

NormalizationMode = Literal[
    "unit",
    "imagenet",
]


INTERPOLATION_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Validated image preprocessing configuration."""

    target_width: int = 224
    target_height: int = 224
    interpolation: InterpolationMode = "bilinear"
    padding_color: tuple[int, int, int] = (0, 0, 0)
    normalization: NormalizationMode = "imagenet"
    max_input_pixels: int = 25_000_000

    def __post_init__(self) -> None:
        if self.target_width <= 0:
            raise ValueError("target_width must be positive.")

        if self.target_height <= 0:
            raise ValueError("target_height must be positive.")

        if self.interpolation not in INTERPOLATION_MAP:
            raise ValueError("Unsupported interpolation mode.")

        if self.normalization not in {"unit", "imagenet"}:
            raise ValueError("Unsupported normalization mode.")

        if len(self.padding_color) != 3:
            raise ValueError("padding_color must contain exactly three values.")

        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 255
            for value in self.padding_color
        ):
            raise ValueError("padding_color values must be integers from 0 to 255.")

        if self.max_input_pixels <= 0:
            raise ValueError("max_input_pixels must be positive.")


@dataclass(frozen=True)
class PreprocessingMetadata:
    """Geometry and transformation information."""

    original_width: int
    original_height: int
    oriented_width: int
    oriented_height: int
    resized_width: int
    resized_height: int
    target_width: int
    target_height: int
    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int
    normalization: str


@dataclass
class PreprocessingResult:
    """Model-ready image, mask, preview, and metadata."""

    image_tensor: torch.Tensor
    valid_mask: torch.Tensor
    preview_rgb: np.ndarray
    metadata: PreprocessingMetadata


def load_preprocessing_config(
    config_path: Path,
) -> PreprocessingConfig:
    """Load and validate preprocessing settings from YAML."""

    if not config_path.is_file():
        raise FileNotFoundError(f"Preprocessing config not found: {config_path}")

    with config_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Preprocessing configuration must be a YAML mapping.")

    if raw.get("schema_version") != "1.0":
        raise ValueError("Unsupported preprocessing schema version.")

    image = raw.get("image")
    normalization = raw.get("normalization")
    safety = raw.get("safety")

    if not all(isinstance(section, dict) for section in (image, normalization, safety)):
        raise ValueError("Missing or invalid configuration sections.")

    required_image = {
        "target_width",
        "target_height",
        "interpolation",
        "padding_color",
    }

    if not required_image.issubset(image):
        raise ValueError("Missing required image configuration fields.")

    if "mode" not in normalization:
        raise ValueError("Missing normalization.mode.")

    if "max_input_pixels" not in safety:
        raise ValueError("Missing safety.max_input_pixels.")

    return PreprocessingConfig(
        target_width=image["target_width"],
        target_height=image["target_height"],
        interpolation=image["interpolation"],
        padding_color=tuple(image["padding_color"]),
        normalization=normalization["mode"],
        max_input_pixels=safety["max_input_pixels"],
    )


class ImagePreprocessor:
    """Reusable preprocessing for training and inference."""

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:
        self.config = config

    def process(
        self,
        image_path: Path,
    ) -> PreprocessingResult:
        """Load and preprocess one image."""

        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with Image.open(image_path) as source:
                original_width, original_height = source.size

                if original_width * original_height > self.config.max_input_pixels:
                    raise ValueError("Image exceeds configured pixel limit.")

                oriented = ImageOps.exif_transpose(source)

                rgb = oriented.convert("RGB")

                oriented_width, oriented_height = rgb.size

                if oriented_width * oriented_height > self.config.max_input_pixels:
                    raise ValueError("Oriented image exceeds pixel limit.")

                result = self._transform(
                    rgb=rgb,
                    original_size=(
                        original_width,
                        original_height,
                    ),
                )

                return result

        except (
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise ValueError("Unable to decode input image.") from exc

    def _transform(
        self,
        rgb: Image.Image,
        original_size: tuple[int, int],
    ) -> PreprocessingResult:
        """Resize, pad, normalize, and create the valid mask."""

        width, height = rgb.size

        target_width = self.config.target_width
        target_height = self.config.target_height

        scale = min(
            target_width / width,
            target_height / height,
        )

        resized_width = max(
            1,
            min(
                target_width,
                int(math.floor(width * scale + 0.5)),
            ),
        )

        resized_height = max(
            1,
            min(
                target_height,
                int(math.floor(height * scale + 0.5)),
            ),
        )

        resized = rgb.resize(
            (resized_width, resized_height),
            resample=INTERPOLATION_MAP[self.config.interpolation],
        )

        padding_left = (target_width - resized_width) // 2

        padding_top = (target_height - resized_height) // 2

        padding_right = target_width - resized_width - padding_left

        padding_bottom = target_height - resized_height - padding_top

        canvas = Image.new(
            "RGB",
            (target_width, target_height),
            color=self.config.padding_color,
        )

        canvas.paste(
            resized,
            (padding_left, padding_top),
        )

        preview_rgb = np.asarray(
            canvas,
            dtype=np.uint8,
        ).copy()

        mask_array = np.zeros(
            (target_height, target_width),
            dtype=np.float32,
        )

        mask_array[
            padding_top : padding_top + resized_height,
            padding_left : padding_left + resized_width,
        ] = 1.0

        image_array = preview_rgb.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())

        if self.config.normalization == "imagenet":
            mean = torch.tensor(
                IMAGENET_MEAN,
                dtype=torch.float32,
            ).view(3, 1, 1)

            std = torch.tensor(
                IMAGENET_STD,
                dtype=torch.float32,
            ).view(3, 1, 1)

            image_tensor = (image_tensor - mean) / std

        valid_mask = torch.from_numpy(mask_array.copy()).unsqueeze(0)

        metadata = PreprocessingMetadata(
            original_width=original_size[0],
            original_height=original_size[1],
            oriented_width=width,
            oriented_height=height,
            resized_width=resized_width,
            resized_height=resized_height,
            target_width=target_width,
            target_height=target_height,
            padding_left=padding_left,
            padding_top=padding_top,
            padding_right=padding_right,
            padding_bottom=padding_bottom,
            normalization=self.config.normalization,
        )

        return PreprocessingResult(
            image_tensor=image_tensor,
            valid_mask=valid_mask,
            preview_rgb=preview_rgb,
            metadata=metadata,
        )

"""Image exploratory data analysis for NeoJaundice."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageStatistics:
    """Image-level descriptive statistics."""

    relative_path: str
    body_region_folder: str
    width: int
    height: int
    aspect_ratio: float
    brightness: float
    contrast: float
    blur_score: float
    red_mean: float
    green_mean: float
    blue_mean: float
    saturation_mean: float
    overexposed_fraction: float
    underexposed_fraction: float


def discover_images(dataset_root: Path) -> list[Path]:
    """Discover supported image files without following file symlinks."""

    if dataset_root.is_symlink():
        raise ValueError("Dataset root must not be a symbolic link.")

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_root}")

    return sorted(
        path
        for path in dataset_root.rglob("*")
        if (
            path.is_file()
            and not path.is_symlink()
            and not any(
                parent.is_symlink()
                for parent in path.relative_to(dataset_root).parents
                if parent != Path(".")
            )
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )


def detect_body_region(
    image_path: Path,
    dataset_root: Path,
) -> str:
    """Detect anatomical region using the nearest matching folder.

    Descriptive parent directories containing both anatomical
    names are ignored.
    """

    relative_path = image_path.relative_to(dataset_root)

    # Exclude the image filename and inspect directories
    # from the nearest parent toward the dataset root.
    directories = reversed(relative_path.parts[:-1])

    for directory in directories:
        normalized = directory.casefold().strip()

        # Recognize explicit anatomical folder names.
        if normalized in {
            "forehead",
            "forehead_images",
            "forehead_photos",
        }:
            return "forehead"

        if normalized in {
            "sternum",
            "sternum_images",
            "sternum_photos",
        }:
            return "sternum"

    return "unknown"


def analyze_image(
    image_path: Path,
    dataset_root: Path,
) -> ImageStatistics:
    """Extract descriptive image metrics without modifying the image."""

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(f"Unable to decode image: {image_path.name}")

    height, width = image.shape[:2]

    if height == 0 or width == 0:
        raise ValueError("Image dimensions must be positive.")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    relative_path = image_path.relative_to(dataset_root)

    region_folder = detect_body_region(
        image_path,
        dataset_root,
    )

    return ImageStatistics(
        relative_path=relative_path.as_posix(),
        body_region_folder=region_folder,
        width=width,
        height=height,
        aspect_ratio=float(width / height),
        brightness=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        blur_score=float(np.var(laplacian)),
        red_mean=float(np.mean(rgb[:, :, 0])),
        green_mean=float(np.mean(rgb[:, :, 1])),
        blue_mean=float(np.mean(rgb[:, :, 2])),
        saturation_mean=float(np.mean(hsv[:, :, 1])),
        overexposed_fraction=float(np.mean(gray >= 245)),
        underexposed_fraction=float(np.mean(gray <= 10)),
    )


def analyze_dataset(
    dataset_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Analyze all images and collect non-sensitive failure information."""

    image_paths = discover_images(dataset_root)

    if not image_paths:
        raise ValueError("No supported images found in dataset directory.")

    records: list[dict] = []
    failures: list[str] = []

    for index, image_path in enumerate(image_paths, start=1):
        try:
            statistics = analyze_image(
                image_path,
                dataset_root,
            )

            records.append(asdict(statistics))

        except (ValueError, cv2.error, OSError):
            relative_path = image_path.relative_to(dataset_root).as_posix()

            failures.append(relative_path)

            LOGGER.warning(
                "Image analysis failed for file index %s",
                index,
            )

        if index % 100 == 0:
            LOGGER.info(
                "Processed %s/%s images",
                index,
                len(image_paths),
            )

    if not records:
        raise ValueError("No images could be analyzed successfully.")

    dataframe = pd.DataFrame.from_records(records)

    return dataframe, failures


def build_dataset_summary(
    dataframe: pd.DataFrame,
    failures: list[str],
) -> dict:
    """Create dataset-level descriptive statistics."""

    numeric_columns = [
        "width",
        "height",
        "aspect_ratio",
        "brightness",
        "contrast",
        "blur_score",
        "red_mean",
        "green_mean",
        "blue_mean",
        "saturation_mean",
        "overexposed_fraction",
        "underexposed_fraction",
    ]

    metric_summary = {}

    for column in numeric_columns:
        series = dataframe[column]

        metric_summary[column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    region_counts = (
        dataframe["body_region_folder"].value_counts().sort_index().to_dict()
    )

    return {
        "analyzed_images": int(len(dataframe)),
        "failed_images": len(failures),
        "region_folder_counts": {
            str(key): int(value) for key, value in region_counts.items()
        },
        "metrics": metric_summary,
        "limitations": [
            "No bilirubin reference labels were analyzed.",
            "Folder names are descriptive, not verified clinical labels.",
            "Blur and exposure indicators are not clinical quality thresholds.",
            "Color statistics are affected by camera and lighting conditions.",
        ],
    }

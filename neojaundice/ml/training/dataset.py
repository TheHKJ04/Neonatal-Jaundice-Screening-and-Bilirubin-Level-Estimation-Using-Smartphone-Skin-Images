"""Patient-split image datasets for NeoJaundice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_transform(mode: str, training: bool = False) -> transforms.Compose:
    """Use mild geometry-only augmentation; do not alter diagnostic colors."""
    if mode not in {"resize", "letterbox", "augment"}:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    if mode == "letterbox":
        spatial = [
            transforms.Lambda(
                lambda im: ImageOps.pad(
                    im,
                    (224, 224),
                    method=Image.Resampling.BILINEAR,
                    color=(0, 0, 0),
                    centering=(0.5, 0.5),
                )
            )
        ]
    else:
        spatial = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
        ]
    if mode == "augment" and training:
        spatial.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=5, interpolation=InterpolationMode.BILINEAR
                ),
            ]
        )
    return transforms.Compose(
        spatial + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


class NeoJaundiceDataset(Dataset):
    """Read a prepared split manifest without duplicating image files."""

    def __init__(
        self,
        manifest: str | Path,
        image_root: str | Path,
        task: str,
        preprocessing: str = "resize",
        training: bool = False,
        exclude_zero: bool = False,
    ):
        self.manifest = pd.read_csv(
            manifest, dtype={"patient_id": str, "image_id": str}
        )
        self.root = Path(image_root).resolve()
        self.task = task
        self.transform = make_transform(preprocessing, training)
        if task not in {"regression", "classification"}:
            raise ValueError("task must be regression or classification")
        required = {
            "image_id",
            "patient_id",
            "image_path",
            "bilirubin_value",
            "treatment_label",
        }
        if not required.issubset(self.manifest.columns):
            raise ValueError(
                f"Missing manifest columns: {sorted(required - set(self.manifest.columns))}"
            )
        if exclude_zero and task == "regression":
            self.manifest = self.manifest.loc[
                pd.to_numeric(self.manifest["bilirubin_value"], errors="coerce") > 0
            ].copy()
        target = "bilirubin_value" if task == "regression" else "treatment_label"
        values = pd.to_numeric(self.manifest[target], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError("Targets contain missing or non-finite values")
        if task == "regression" and (values < 0).any():
            raise ValueError("Negative bilirubin value")
        if task == "classification" and not np.isin(values, [0, 1]).all():
            raise ValueError("Classification targets must be 0 or 1")
        if self.manifest.empty:
            raise ValueError("Dataset has no records")
        self.targets = values.astype(np.float32)
        self.paths: list[Path] = []
        for rel in self.manifest["image_path"].astype(str):
            candidate = Path(rel)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Unsafe relative image path: {rel}")
            path = (self.root / candidate).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Image missing or outside dataset root: {rel}")
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        with Image.open(self.paths[index]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            tensor = self.transform(image)
        row = self.manifest.iloc[index]
        return {
            "image": tensor,
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "patient_id": str(row["patient_id"]),
            "image_id": str(row["image_id"]),
        }

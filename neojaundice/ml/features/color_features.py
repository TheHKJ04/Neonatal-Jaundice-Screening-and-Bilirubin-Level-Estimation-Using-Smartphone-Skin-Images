"""Color feature extraction from geometric skin ROIs."""

from __future__ import annotations

import cv2
import numpy as np
import torch

COLOR_SPACES = {
    "rgb": None,
    "hsv": cv2.COLOR_RGB2HSV,
    "ycrcb": cv2.COLOR_RGB2YCrCb,
    "lab": cv2.COLOR_RGB2LAB,
}

CHANNEL_NAMES = {
    "rgb": ("r", "g", "b"),
    "hsv": ("h", "s", "v"),
    "ycrcb": ("y", "cr", "cb"),
    "lab": ("l", "a", "b"),
}


def extract_color_features(
    image_rgb: np.ndarray,
    roi_mask: torch.Tensor,
) -> dict[str, float]:
    """Extract color statistics using pixels inside the ROI."""

    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a NumPy array.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    if image_rgb.dtype != np.uint8:
        raise ValueError("RGB image must have dtype uint8.")

    if not isinstance(roi_mask, torch.Tensor):
        raise TypeError("roi_mask must be a PyTorch tensor.")

    if roi_mask.ndim != 3 or roi_mask.shape[0] != 1:
        raise ValueError("ROI mask must have shape (1, H, W).")

    if not torch.isfinite(roi_mask).all().item():
        raise ValueError("ROI mask contains non-finite values.")

    if (
        not torch.logical_or(
            roi_mask == 0,
            roi_mask == 1,
        )
        .all()
        .item()
    ):
        raise ValueError("ROI mask must be binary.")

    mask = roi_mask.detach().cpu().numpy()[0].astype(bool)

    if mask.shape != image_rgb.shape[:2]:
        raise ValueError("ROI mask and image dimensions do not match.")

    if not np.any(mask):
        raise ValueError("ROI mask is empty.")

    features: dict[str, float] = {
        "roi_pixel_count": float(np.count_nonzero(mask)),
    }

    for color_space, conversion_code in COLOR_SPACES.items():
        if conversion_code is None:
            converted = image_rgb
        else:
            converted = cv2.cvtColor(image_rgb, conversion_code)

        pixels = converted[mask].astype(np.float64)

        for channel_index, channel_name in enumerate(CHANNEL_NAMES[color_space]):
            values = pixels[:, channel_index]
            prefix = f"{color_space}_{channel_name}"

            features[f"{prefix}_mean"] = float(np.mean(values))
            features[f"{prefix}_std"] = float(np.std(values))
            features[f"{prefix}_median"] = float(np.median(values))
            features[f"{prefix}_p10"] = float(np.percentile(values, 10))
            features[f"{prefix}_p90"] = float(np.percentile(values, 90))

    return features

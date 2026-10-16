"""Pixel-level agreement metrics for candidate segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MaskAgreementResult:
    """Pixel-level agreement between two candidate masks."""

    intersection_pixels: int
    union_pixels: int

    ycrcb_only_pixels: int
    hsv_only_pixels: int

    both_rejected_pixels: int
    geometric_roi_pixels: int

    iou: float | None
    dice: float | None

    disagreement_rate: float

    both_masks_empty: bool


def _to_binary_numpy(
    mask: torch.Tensor,
    name: str,
) -> np.ndarray:
    """Validate a binary tensor and convert it to a NumPy mask."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor.")

    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"{name} must have shape (1, H, W).")

    if not torch.isfinite(mask).all().item():
        raise ValueError(f"{name} contains non-finite values.")

    binary = torch.logical_or(
        mask == 0,
        mask == 1,
    )

    if not binary.all().item():
        raise ValueError(f"{name} must contain only 0 and 1.")

    return mask.detach().cpu().numpy()[0].astype(bool)


def calculate_mask_agreement(
    ycrcb_mask: torch.Tensor,
    hsv_mask: torch.Tensor,
    geometric_roi_mask: torch.Tensor,
) -> MaskAgreementResult:
    """Calculate pixel agreement inside a geometric ROI."""

    ycrcb = _to_binary_numpy(
        ycrcb_mask,
        "ycrcb_mask",
    )

    hsv = _to_binary_numpy(
        hsv_mask,
        "hsv_mask",
    )

    roi = _to_binary_numpy(
        geometric_roi_mask,
        "geometric_roi_mask",
    )

    if not (ycrcb.shape == hsv.shape == roi.shape):
        raise ValueError("All masks must have identical dimensions.")

    geometric_pixels = int(np.count_nonzero(roi))

    if geometric_pixels == 0:
        raise ValueError("Geometric ROI cannot be empty.")

    if np.any(ycrcb & ~roi):
        raise ValueError("YCrCb mask extends outside geometric ROI.")

    if np.any(hsv & ~roi):
        raise ValueError("HSV mask extends outside geometric ROI.")

    intersection = ycrcb & hsv
    union = ycrcb | hsv

    ycrcb_only = ycrcb & ~hsv
    hsv_only = hsv & ~ycrcb

    both_rejected = roi & ~union

    intersection_pixels = int(np.count_nonzero(intersection))

    union_pixels = int(np.count_nonzero(union))

    ycrcb_only_pixels = int(np.count_nonzero(ycrcb_only))

    hsv_only_pixels = int(np.count_nonzero(hsv_only))

    both_rejected_pixels = int(np.count_nonzero(both_rejected))

    ycrcb_pixels = int(np.count_nonzero(ycrcb))

    hsv_pixels = int(np.count_nonzero(hsv))

    both_empty = union_pixels == 0

    if both_empty:
        iou = None
        dice = None
    else:
        iou = intersection_pixels / union_pixels

        dice = 2 * intersection_pixels / (ycrcb_pixels + hsv_pixels)

    disagreement_rate = (ycrcb_only_pixels + hsv_only_pixels) / geometric_pixels

    return MaskAgreementResult(
        intersection_pixels=intersection_pixels,
        union_pixels=union_pixels,
        ycrcb_only_pixels=ycrcb_only_pixels,
        hsv_only_pixels=hsv_only_pixels,
        both_rejected_pixels=both_rejected_pixels,
        geometric_roi_pixels=geometric_pixels,
        iou=iou,
        dice=dice,
        disagreement_rate=disagreement_rate,
        both_masks_empty=both_empty,
    )


def create_agreement_overlay(
    preview_rgb: np.ndarray,
    ycrcb_mask: torch.Tensor,
    hsv_mask: torch.Tensor,
    geometric_roi_mask: torch.Tensor,
) -> np.ndarray:
    """Visualize method agreement and disagreement.

    Green: selected by both methods.
    Red: selected only by YCrCb.
    Blue: selected only by HSV.
    Original RGB: rejected by both methods.
    """

    if (
        preview_rgb.ndim != 3
        or preview_rgb.shape[2] != 3
        or preview_rgb.dtype != np.uint8
    ):
        raise ValueError("Expected uint8 RGB preview.")

    ycrcb = _to_binary_numpy(
        ycrcb_mask,
        "ycrcb_mask",
    )

    hsv = _to_binary_numpy(
        hsv_mask,
        "hsv_mask",
    )

    roi = _to_binary_numpy(
        geometric_roi_mask,
        "geometric_roi_mask",
    )

    if not (ycrcb.shape == hsv.shape == roi.shape == preview_rgb.shape[:2]):
        raise ValueError("Image and mask dimensions do not match.")

    if np.any((ycrcb | hsv) & ~roi):
        raise ValueError("Candidate masks extend outside geometric ROI.")

    overlay = preview_rgb.copy()

    both = ycrcb & hsv
    ycrcb_only = ycrcb & ~hsv
    hsv_only = hsv & ~ycrcb

    overlay[both] = [0, 255, 0]
    overlay[ycrcb_only] = [255, 0, 0]
    overlay[hsv_only] = [0, 0, 255]

    return overlay

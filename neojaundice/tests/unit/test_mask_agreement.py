"""Tests for pixel-level mask agreement."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.evaluation.mask_agreement import (
    calculate_mask_agreement,
    create_agreement_overlay,
)


def make_mask(
    values: list[list[int]],
) -> torch.Tensor:
    return torch.tensor(
        [values],
        dtype=torch.float32,
    )


@pytest.fixture
def full_roi():
    return torch.ones(
        (1, 2, 2),
        dtype=torch.float32,
    )


def test_identical_masks(full_roi):
    mask = make_mask(
        [
            [1, 1],
            [0, 0],
        ]
    )

    result = calculate_mask_agreement(
        mask,
        mask,
        full_roi,
    )

    assert result.iou == pytest.approx(1.0)
    assert result.dice == pytest.approx(1.0)
    assert result.disagreement_rate == 0.0


def test_partial_overlap(full_roi):
    ycrcb = make_mask(
        [
            [1, 1],
            [0, 0],
        ]
    )

    hsv = make_mask(
        [
            [1, 0],
            [1, 0],
        ]
    )

    result = calculate_mask_agreement(
        ycrcb,
        hsv,
        full_roi,
    )

    assert result.intersection_pixels == 1
    assert result.union_pixels == 3

    assert result.iou == pytest.approx(1 / 3)

    assert result.dice == pytest.approx(0.5)

    assert result.disagreement_rate == pytest.approx(0.5)


def test_disjoint_masks(full_roi):
    ycrcb = make_mask(
        [
            [1, 0],
            [0, 0],
        ]
    )

    hsv = make_mask(
        [
            [0, 1],
            [0, 0],
        ]
    )

    result = calculate_mask_agreement(
        ycrcb,
        hsv,
        full_roi,
    )

    assert result.iou == 0.0
    assert result.dice == 0.0


def test_both_masks_empty(full_roi):
    empty = torch.zeros_like(full_roi)

    result = calculate_mask_agreement(
        empty,
        empty,
        full_roi,
    )

    assert result.iou is None
    assert result.dice is None

    assert result.both_masks_empty is True
    assert result.disagreement_rate == 0.0


def test_one_mask_empty(full_roi):
    empty = torch.zeros_like(full_roi)

    result = calculate_mask_agreement(
        empty,
        full_roi,
        full_roi,
    )

    assert result.iou == 0.0
    assert result.dice == 0.0

    assert result.disagreement_rate == 1.0


def test_identical_full_masks(full_roi):
    result = calculate_mask_agreement(
        full_roi,
        full_roi,
        full_roi,
    )

    assert result.iou == 1.0
    assert result.dice == 1.0

    assert result.both_rejected_pixels == 0


def test_both_rejected_pixels(full_roi):
    mask = make_mask(
        [
            [1, 0],
            [0, 0],
        ]
    )

    result = calculate_mask_agreement(
        mask,
        mask,
        full_roi,
    )

    assert result.both_rejected_pixels == 3


def test_ycrcb_only_pixels(full_roi):
    ycrcb = make_mask(
        [
            [1, 1],
            [0, 0],
        ]
    )

    hsv = make_mask(
        [
            [1, 0],
            [0, 0],
        ]
    )

    result = calculate_mask_agreement(
        ycrcb,
        hsv,
        full_roi,
    )

    assert result.ycrcb_only_pixels == 1
    assert result.hsv_only_pixels == 0


def test_hsv_only_pixels(full_roi):
    ycrcb = make_mask(
        [
            [1, 0],
            [0, 0],
        ]
    )

    hsv = make_mask(
        [
            [1, 1],
            [0, 0],
        ]
    )

    result = calculate_mask_agreement(
        ycrcb,
        hsv,
        full_roi,
    )

    assert result.hsv_only_pixels == 1
    assert result.ycrcb_only_pixels == 0


def test_invalid_shape(full_roi):
    invalid = torch.zeros(
        (2, 2),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        calculate_mask_agreement(
            invalid,
            full_roi,
            full_roi,
        )


def test_mismatched_dimensions(full_roi):
    smaller = torch.zeros(
        (1, 1, 1),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        calculate_mask_agreement(
            smaller,
            full_roi,
            full_roi,
        )


def test_nonbinary_mask(full_roi):
    invalid = full_roi.clone()
    invalid[0, 0, 0] = 0.5

    with pytest.raises(ValueError):
        calculate_mask_agreement(
            invalid,
            full_roi,
            full_roi,
        )


def test_nonfinite_mask(full_roi):
    invalid = full_roi.clone()
    invalid[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        calculate_mask_agreement(
            invalid,
            full_roi,
            full_roi,
        )


def test_empty_geometric_roi():
    empty = torch.zeros(
        (1, 2, 2),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError):
        calculate_mask_agreement(
            empty,
            empty,
            empty,
        )


def test_mask_outside_roi():
    roi = make_mask(
        [
            [1, 0],
            [0, 0],
        ]
    )

    invalid = make_mask(
        [
            [1, 1],
            [0, 0],
        ]
    )

    with pytest.raises(ValueError):
        calculate_mask_agreement(
            invalid,
            roi,
            roi,
        )


def test_overlay_shape(full_roi):
    image = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )

    overlay = create_agreement_overlay(
        image,
        full_roi,
        full_roi,
        full_roi,
    )

    assert overlay.shape == (
        2,
        2,
        3,
    )


def test_overlay_agreement_color(full_roi):
    image = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )

    overlay = create_agreement_overlay(
        image,
        full_roi,
        full_roi,
        full_roi,
    )

    assert np.array_equal(
        overlay[0, 0],
        [0, 255, 0],
    )


def test_overlay_ycrcb_only_color(full_roi):
    image = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )

    ycrcb = make_mask(
        [
            [1, 0],
            [0, 0],
        ]
    )

    hsv = torch.zeros_like(full_roi)

    overlay = create_agreement_overlay(
        image,
        ycrcb,
        hsv,
        full_roi,
    )

    assert np.array_equal(
        overlay[0, 0],
        [255, 0, 0],
    )


def test_overlay_hsv_only_color(full_roi):
    image = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )

    ycrcb = torch.zeros_like(full_roi)

    hsv = make_mask(
        [
            [1, 0],
            [0, 0],
        ]
    )

    overlay = create_agreement_overlay(
        image,
        ycrcb,
        hsv,
        full_roi,
    )

    assert np.array_equal(
        overlay[0, 0],
        [0, 0, 255],
    )


def test_overlay_does_not_modify_input(full_roi):
    image = np.full(
        (2, 2, 3),
        100,
        dtype=np.uint8,
    )

    original = image.copy()

    create_agreement_overlay(
        image,
        full_roi,
        full_roi,
        full_roi,
    )

    assert np.array_equal(
        image,
        original,
    )

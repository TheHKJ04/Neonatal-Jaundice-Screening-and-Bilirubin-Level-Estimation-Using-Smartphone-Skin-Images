"""Tests for NeoJaundice adaptation and patient splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.datasets.neojaundice_adapter import (
    create_patient_splits,
    prepare_manifest,
)


def make_manifest(patient_count: int = 100) -> pd.DataFrame:
    records = []

    for patient_index in range(patient_count):
        patient_id = f"{patient_index:04d}"
        treatment = patient_index % 2

        for image_index in range(3):
            records.append(
                {
                    "image_id": (f"{patient_id}-{image_index + 1}.jpg"),
                    "patient_id": patient_id,
                    "treatment_label": treatment,
                    "bilirubin_value": float(patient_index + 1),
                }
            )

    return pd.DataFrame(records)


def test_patient_splits_have_no_overlap():
    manifest = make_manifest()

    splits = create_patient_splits(manifest)

    groups = {name: set(df["patient_id"]) for name, df in splits.items()}

    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_all_images_are_preserved():
    manifest = make_manifest()

    splits = create_patient_splits(manifest)

    assert sum(len(dataframe) for dataframe in splits.values()) == len(manifest)


def test_split_is_reproducible():
    manifest = make_manifest()

    first = create_patient_splits(
        manifest,
        random_seed=42,
    )

    second = create_patient_splits(
        manifest,
        random_seed=42,
    )

    for name in ("train", "val", "test"):
        assert first[name]["image_id"].tolist() == (second[name]["image_id"].tolist())


def test_invalid_split_fractions():
    manifest = make_manifest()

    with pytest.raises(ValueError):
        create_patient_splits(
            manifest,
            train_fraction=0.8,
            validation_fraction=0.2,
            test_fraction=0.2,
        )


def test_conflicting_patient_labels_rejected():
    manifest = make_manifest()

    manifest.loc[0, "treatment_label"] = 1

    with pytest.raises(ValueError):
        create_patient_splits(manifest)


def test_missing_metadata_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_manifest(tmp_path)


def test_missing_image_rejected(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    dataframe = pd.DataFrame(
        {
            "patient_id": ["0001"],
            "image_idx": ["missing.jpg"],
            "gender": ["F"],
            "gestational_age": [40],
            "age(day)": [5.0],
            "weight": [3000],
            "blood(mg/dL)": [10.0],
            "Treatment": [0],
        }
    )

    dataframe.to_csv(
        tmp_path / "chd_jaundice_published_2.csv",
        index=False,
    )

    with pytest.raises(FileNotFoundError):
        prepare_manifest(tmp_path)


def test_zero_bilirubin_is_flagged(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    (image_dir / "0001-1.jpg").write_bytes(b"test")

    dataframe = pd.DataFrame(
        {
            "patient_id": ["0001"],
            "image_idx": ["0001-1.jpg"],
            "gender": ["F"],
            "gestational_age": [40],
            "age(day)": [5.0],
            "weight": [3000],
            "blood(mg/dL)": [0.0],
            "Treatment": [0],
        }
    )

    dataframe.to_csv(
        tmp_path / "chd_jaundice_published_2.csv",
        index=False,
    )

    manifest = prepare_manifest(tmp_path)

    assert bool(manifest.loc[0, "zero_bilirubin_review"])

    assert manifest.loc[0, "bilirubin_value"] == 0.0


def test_missing_age_is_preserved(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    (image_dir / "0001-1.jpg").write_bytes(b"test")

    dataframe = pd.DataFrame(
        {
            "patient_id": ["0001"],
            "image_idx": ["0001-1.jpg"],
            "gender": ["F"],
            "gestational_age": [40],
            "age(day)": [None],
            "weight": [3000],
            "blood(mg/dL)": [10.0],
            "Treatment": [0],
        }
    )

    dataframe.to_csv(
        tmp_path / "chd_jaundice_published_2.csv",
        index=False,
    )

    manifest = prepare_manifest(tmp_path)

    assert pd.isna(manifest.loc[0, "postnatal_age_days"])

    assert bool(manifest.loc[0, "missing_age_review"])

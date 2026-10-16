"""Unit tests for labeled dataset validation."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.datasets.label_validator import (
    LabeledDatasetValidator,
)


@pytest.fixture
def valid_dataset(tmp_path):
    image_directory = tmp_path / "images"
    image_directory.mkdir()

    (image_directory / "001.jpg").write_bytes(b"test-image-placeholder")

    dataframe = pd.DataFrame(
        [
            {
                "image_id": "IMG001",
                "patient_id": "P001",
                "image_path": "images/001.jpg",
                "body_region": "forehead",
                "bilirubin_value": 12.4,
                "bilirubin_unit": "mg/dL",
                "measurement_type": "TSB",
            }
        ]
    )

    return dataframe, tmp_path


def test_valid_dataset_passes(valid_dataset):
    dataframe, root = valid_dataset

    report = LabeledDatasetValidator(root).validate(dataframe)

    assert report.passed
    assert report.valid_records == 1


def test_missing_column_fails(valid_dataset):
    dataframe, root = valid_dataset

    dataframe = dataframe.drop(columns=["bilirubin_value"])

    report = LabeledDatasetValidator(root).validate(dataframe)

    assert not report.passed


def test_missing_image_fails(valid_dataset):
    dataframe, root = valid_dataset

    dataframe.loc[
        0,
        "image_path",
    ] = "images/missing.jpg"

    report = LabeledDatasetValidator(root).validate(dataframe)

    assert not report.passed


def test_negative_bilirubin_fails(valid_dataset):
    dataframe, root = valid_dataset

    dataframe.loc[
        0,
        "bilirubin_value",
    ] = -1

    report = LabeledDatasetValidator(root).validate(dataframe)

    assert not report.passed


def test_invalid_unit_fails(valid_dataset):
    dataframe, root = valid_dataset

    dataframe.loc[
        0,
        "bilirubin_unit",
    ] = "unknown"

    report = LabeledDatasetValidator(root).validate(dataframe)

    assert not report.passed


def test_invalid_measurement_type_fails(valid_dataset):
    dataframe, root = valid_dataset

    dataframe.loc[
        0,
        "measurement_type",
    ] = "UNKNOWN"

    report = LabeledDatasetValidator(root).validate(dataframe)

    assert not report.passed


def test_missing_patient_id_fails(valid_dataset):
    dataframe, root = valid_dataset

    dataframe.loc[
        0,
        "patient_id",
    ] = None

    report = LabeledDatasetValidator(root).validate(dataframe)

    assert not report.passed


def test_duplicate_image_id_fails(valid_dataset):
    dataframe, root = valid_dataset

    duplicated = pd.concat(
        [dataframe, dataframe],
        ignore_index=True,
    )

    report = LabeledDatasetValidator(root).validate(duplicated)

    assert not report.passed


def test_path_traversal_fails(valid_dataset):
    dataframe, root = valid_dataset

    dataframe.loc[
        0,
        "image_path",
    ] = "../outside.jpg"

    report = LabeledDatasetValidator(root).validate(dataframe)

    assert not report.passed


def test_empty_dataset_fails(tmp_path):
    dataframe = pd.DataFrame(
        columns=[
            "image_id",
            "patient_id",
            "image_path",
            "body_region",
            "bilirubin_value",
            "bilirubin_unit",
            "measurement_type",
        ]
    )

    report = LabeledDatasetValidator(tmp_path).validate(dataframe)

    assert not report.passed

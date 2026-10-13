"""
Unit tests for NeoJaundice dataset metadata schemas.
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ml.datasets.schemas import (
    BilirubinType,
    DatasetSample,
)

UTC = timezone.utc


def make_valid_sample(**overrides):
    """Create a synthetic valid metadata record."""

    sample = {
        "sample_id": "S001",
        "patient_id": "P001",
        "image_path": "images/sample_001.jpg",
        "captured_at": datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
        "bilirubin_measured_at": datetime(2026, 1, 10, 10, 30, tzinfo=UTC),
        "bilirubin_type": "TSB",
        "bilirubin_mg_dl": 12.8,
        "age_hours": 72.0,
        "gestational_age_weeks": 38.0,
        "body_region": "forehead",
    }

    sample.update(overrides)

    return sample


def test_valid_sample():
    """Valid research metadata should be accepted."""

    sample = DatasetSample.model_validate(make_valid_sample())

    assert sample.sample_id == "S001"

    assert sample.patient_id == "P001"

    assert sample.bilirubin_mg_dl == 12.8


def test_negative_bilirubin_rejected():
    """Negative bilirubin measurements are invalid."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(make_valid_sample(bilirubin_mg_dl=-1.0))


def test_negative_age_rejected():
    """Negative postnatal age is invalid."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(make_valid_sample(age_hours=-2.0))


def test_zero_gestational_age_rejected():
    """Gestational age must be positive."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(make_valid_sample(gestational_age_weeks=0))


def test_invalid_measurement_type_rejected():
    """Unknown measurement types are rejected."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(make_valid_sample(bilirubin_type="UNKNOWN"))


def test_parent_path_rejected():
    """Parent directory traversal is rejected."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(make_valid_sample(image_path="../secret.jpg"))


def test_absolute_path_rejected():
    """Absolute image paths are rejected."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(make_valid_sample(image_path="/private/image.jpg"))


def test_naive_timestamp_rejected():
    """Timestamps without timezone information are rejected."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(
            make_valid_sample(captured_at=datetime(2026, 1, 10, 10, 0))
        )


def test_pairing_interval():
    """Verify pairing interval calculation."""

    sample = DatasetSample.model_validate(make_valid_sample())

    assert sample.pairing_interval_minutes == 30.0


def test_pairing_interval_absolute():
    """Pairing interval must be nonnegative."""

    captured_at = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)

    measured_at = captured_at - timedelta(minutes=45)

    sample = DatasetSample.model_validate(
        make_valid_sample(
            captured_at=captured_at,
            bilirubin_measured_at=measured_at,
        )
    )

    assert sample.pairing_interval_minutes == 45.0


def test_tsb_reference_detection():
    """TSB records should be identified correctly."""

    sample = DatasetSample.model_validate(make_valid_sample())

    assert sample.has_tsb_reference is True


def test_tcb_reference_detection():
    """TcB must not be treated as TSB."""

    sample = DatasetSample.model_validate(
        make_valid_sample(bilirubin_type=BilirubinType.TCB)
    )

    assert sample.has_tsb_reference is False


def test_unknown_fields_rejected():
    """Unexpected metadata fields are rejected."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(make_valid_sample(unknown_field="unexpected"))


def test_nan_bilirubin_rejected():
    """NaN target values are rejected."""

    with pytest.raises(ValidationError):
        DatasetSample.model_validate(make_valid_sample(bilirubin_mg_dl=float("nan")))

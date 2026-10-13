"""
Dataset metadata schemas for NeoJaundice.

These schemas validate research dataset records.

They do not establish clinical validity or determine
whether a sample is eligible for model training.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class BilirubinType(str, Enum):
    """Supported reference measurement types."""

    TSB = "TSB"
    TCB = "TcB"


class BodyRegion(str, Enum):
    """Supported image acquisition regions."""

    FOREHEAD = "forehead"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    ARM = "arm"
    LEG = "leg"
    OTHER = "other"


class DatasetSample(BaseModel):
    """Metadata describing one research image sample."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    sample_id: str = Field(
        min_length=1,
        max_length=128,
    )

    patient_id: str = Field(
        min_length=1,
        max_length=128,
    )

    image_path: str = Field(
        min_length=1,
    )

    captured_at: datetime

    bilirubin_measured_at: datetime

    bilirubin_type: BilirubinType

    bilirubin_mg_dl: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    age_hours: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    gestational_age_weeks: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    body_region: BodyRegion

    device_model: str | None = None

    lighting_condition: str | None = None

    acquisition_site: str | None = None

    reference_method: str | None = None

    phototherapy_status: str | None = None

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        """Require a safe, relative, POSIX-style image path."""

        if "\\" in value:
            raise ValueError("image_path must use forward slashes")

        path = PurePosixPath(value)

        if path.is_absolute():
            raise ValueError("image_path must be relative")

        if ".." in path.parts:
            raise ValueError("image_path cannot contain parent traversal")

        if ":" in value:
            raise ValueError("image_path cannot contain a colon")

        if value in {".", ""}:
            raise ValueError("image_path must reference a file")

        return value

    @model_validator(mode="after")
    def validate_timestamps(self) -> DatasetSample:
        """Require compatible timezone-aware timestamps."""

        timestamps = (
            self.captured_at,
            self.bilirubin_measured_at,
        )

        for timestamp in timestamps:
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("timestamps must include timezone information")

        return self

    @property
    def pairing_interval_minutes(self) -> float:
        """Calculate absolute image-reference time difference."""

        difference = self.captured_at - self.bilirubin_measured_at

        return abs(difference.total_seconds()) / 60.0

    @property
    def has_tsb_reference(self) -> bool:
        """Indicate whether the reference measurement is TSB."""

        return self.bilirubin_type == BilirubinType.TSB

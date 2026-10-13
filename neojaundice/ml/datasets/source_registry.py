"""
Dataset source registry for NeoJaundice.

Validates dataset source metadata without downloading
or accessing clinical image files.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class AccessStatus(str, Enum):
    """Dataset access categories."""

    PUBLIC_DOWNLOAD = "public_download"
    PUBLIC_RECORD = "public_record"
    REQUIRES_VERIFICATION = "requires_verification"
    SAMPLE_ONLY = "sample_only"
    RESTRICTED = "restricted"
    EMBARGOED = "embargoed"


class TaskType(str, Enum):
    """Supported research tasks."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DatasetSource(BaseModel):
    """Metadata describing an external dataset source."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    dataset_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    name: str = Field(min_length=3)

    source_url: str

    provider: str = Field(min_length=2)

    access_status: AccessStatus

    license_status: str = Field(min_length=3)

    task_types: list[TaskType] = Field(
        min_length=1,
    )

    reported_patients: int | None = Field(
        default=None,
        gt=0,
    )

    reported_images: int | None = Field(
        default=None,
        gt=0,
    )

    has_numerical_tsb_labels: bool | None = None

    has_patient_identifiers: bool | None = None

    notes: str = Field(min_length=10)

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        """Require an HTTPS source URL."""

        parsed = urlparse(value)

        if parsed.scheme != "https":
            raise ValueError("Dataset source URL must use HTTPS")

        if not parsed.hostname:
            raise ValueError("Dataset source URL must have a hostname")

        if parsed.username or parsed.password:
            raise ValueError("Dataset source URL must not contain credentials")

        return value

    @model_validator(mode="after")
    def validate_task_types(self) -> DatasetSource:
        """Reject duplicate task declarations."""

        if len(self.task_types) != len(set(self.task_types)):
            raise ValueError("task_types cannot contain duplicates")

        return self


class DatasetRegistry(BaseModel):
    """Collection of registered dataset sources."""

    model_config = ConfigDict(extra="forbid")

    registry_version: str = Field(min_length=1)

    datasets: list[DatasetSource] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> DatasetRegistry:
        """Require unique dataset identifiers."""

        identifiers = [dataset.dataset_id for dataset in self.datasets]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Dataset registry contains duplicate dataset IDs")

        return self


def load_dataset_registry(
    config_path: Path,
) -> DatasetRegistry:
    """
    Load and validate the dataset registry.

    This function reads only local YAML metadata.
    It does not download external datasets.
    """

    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset registry not found: {config_path}")

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_data = yaml.safe_load(file)

    if not isinstance(raw_data, dict):
        raise ValueError("Dataset registry must contain a YAML mapping")

    return DatasetRegistry.model_validate(raw_data)

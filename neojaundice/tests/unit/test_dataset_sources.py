"""
Tests for the NeoJaundice dataset source registry.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ml.datasets.source_registry import (
    DatasetRegistry,
    DatasetSource,
    load_dataset_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PATH = PROJECT_ROOT / "configs" / "data" / "sources.yaml"


def make_valid_source(**overrides):
    """Create synthetic dataset source metadata."""

    source = {
        "dataset_id": "example_dataset",
        "name": "Example Research Dataset",
        "source_url": "https://example.org/dataset",
        "provider": "Example Repository",
        "access_status": "public_record",
        "license_status": "unverified",
        "task_types": ["classification"],
        "reported_patients": 100,
        "reported_images": 200,
        "has_numerical_tsb_labels": None,
        "has_patient_identifiers": None,
        "notes": "Synthetic metadata for unit testing.",
    }

    source.update(overrides)

    return source


def test_valid_dataset_source():
    """Valid source metadata should be accepted."""

    source = DatasetSource.model_validate(make_valid_source())

    assert source.dataset_id == "example_dataset"


def test_http_url_rejected():
    """Non-HTTPS source URLs should be rejected."""

    with pytest.raises(ValidationError):
        DatasetSource.model_validate(
            make_valid_source(source_url="http://example.org/dataset")
        )


def test_url_without_hostname_rejected():
    """Source URLs must contain a hostname."""

    with pytest.raises(ValidationError):
        DatasetSource.model_validate(make_valid_source(source_url="https://"))


def test_url_with_credentials_rejected():
    """Credentials must not appear in source URLs."""

    with pytest.raises(ValidationError):
        DatasetSource.model_validate(
            make_valid_source(source_url=("https://user:password@example.org/data"))
        )


def test_negative_patient_count_rejected():
    """Reported patient counts must be positive."""

    with pytest.raises(ValidationError):
        DatasetSource.model_validate(make_valid_source(reported_patients=-5))


def test_duplicate_tasks_rejected():
    """Duplicate research tasks should be rejected."""

    with pytest.raises(ValidationError):
        DatasetSource.model_validate(
            make_valid_source(
                task_types=[
                    "classification",
                    "classification",
                ]
            )
        )


def test_duplicate_dataset_ids_rejected():
    """Dataset identifiers must be unique."""

    source = make_valid_source()

    with pytest.raises(ValidationError):
        DatasetRegistry.model_validate(
            {
                "registry_version": "0.1.0",
                "datasets": [
                    source,
                    source,
                ],
            }
        )


def test_valid_registry():
    """Valid registry should be accepted."""

    registry = DatasetRegistry.model_validate(
        {
            "registry_version": "0.1.0",
            "datasets": [
                make_valid_source(),
            ],
        }
    )

    assert len(registry.datasets) == 1


def test_missing_registry_file(tmp_path: Path):
    """Missing configuration files should raise an error."""

    with pytest.raises(FileNotFoundError):
        load_dataset_registry(tmp_path / "missing.yaml")


def test_invalid_yaml_root(tmp_path: Path):
    """Registry YAML root must be a mapping."""

    registry_path = tmp_path / "sources.yaml"

    registry_path.write_text(
        "- invalid\n- structure\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_dataset_registry(registry_path)


def test_load_valid_registry(tmp_path: Path):
    """Registry should load from YAML."""

    registry_path = tmp_path / "sources.yaml"

    content = {
        "registry_version": "0.1.0",
        "datasets": [
            make_valid_source(),
        ],
    }

    registry_path.write_text(
        yaml.safe_dump(content),
        encoding="utf-8",
    )

    registry = load_dataset_registry(registry_path)

    assert registry.registry_version == "0.1.0"


def test_project_registry_valid():
    """Project dataset registry must remain valid."""

    registry = load_dataset_registry(REGISTRY_PATH)

    assert len(registry.datasets) >= 1

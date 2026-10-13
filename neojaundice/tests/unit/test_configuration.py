"""
Tests for NeoJaundice configuration management.
"""

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from backend.app.core.config import Settings
from scripts.verify_config import CONFIG_DIR


def test_default_application_settings():
    """Verify application settings defaults."""

    settings = Settings(
        _env_file=None,
    )

    assert settings.app_name == "NeoJaundice"
    assert settings.api_port == 8000


def test_custom_api_port():
    """Verify application settings accept valid overrides."""

    settings = Settings(
        _env_file=None,
        api_port=9000,
    )

    assert settings.api_port == 9000


def test_invalid_api_port():
    """Verify invalid API ports are rejected."""

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            api_port=70000,
        )


def test_invalid_environment():
    """Verify unsupported environments are rejected."""

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            app_env="invalid",
        )


def test_hydra_configuration_composition():
    """Verify Hydra configuration files compose."""

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base=None,
    ):
        config = compose(config_name="config")

    assert config.project.name == "neojaundice"

    assert config.model.architecture == "efficientnet_b0"

    assert config.training.epochs == 30


def test_patient_level_split_configuration():
    """Verify patient-level splitting is configured."""

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base=None,
    ):
        config = compose(config_name="config")

    assert config.data.split_strategy == "patient_level"

    total = (
        config.data.split.train + config.data.split.validation + config.data.split.test
    )

    assert total == pytest.approx(1.0)

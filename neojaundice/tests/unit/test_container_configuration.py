# type: ignore
"""
Unit tests for NeoJaundice Docker configuration.
"""

from pathlib import Path

import yaml

from scripts.container_smoke_test import (
    REQUIRED_PACKAGES,
    check_python_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_compose() -> dict:
    """Load the Docker Compose configuration."""

    compose_file = PROJECT_ROOT / "compose.yaml"

    with compose_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def test_dockerfile_exists():
    """Verify the development Dockerfile exists."""

    assert (PROJECT_ROOT / "docker" / "Dockerfile.dev").is_file()


def test_dockerignore_exists():
    """Verify .dockerignore exists."""

    assert (PROJECT_ROOT / ".dockerignore").is_file()


def test_compose_file_exists():
    """Verify compose.yaml exists."""

    assert (PROJECT_ROOT / "compose.yaml").is_file()


def test_required_services_exist():
    """Verify required services are configured."""

    compose = load_compose()

    services = compose["services"]

    assert "app-check" in services
    assert "postgres" in services


def test_postgres_uses_named_volume():
    """Verify PostgreSQL persistence is configured."""

    compose = load_compose()

    postgres = compose["services"]["postgres"]

    assert "postgres-data:/var/lib/postgresql/data" in postgres["volumes"]


def test_postgres_has_healthcheck():
    """Verify PostgreSQL has a health check."""

    compose = load_compose()

    postgres = compose["services"]["postgres"]

    assert "healthcheck" in postgres


def test_python_version():
    """Verify the selected Python version."""

    assert check_python_version()


def test_smoke_test_package_list():
    """Verify smoke test package requirements."""

    assert "numpy" in REQUIRED_PACKAGES
    assert "pytest" in REQUIRED_PACKAGES

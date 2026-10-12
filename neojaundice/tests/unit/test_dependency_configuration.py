"""
Tests for NeoJaundice dependency configuration.
"""

import tomllib

from pathlib import Path
from unittest.mock import patch

from scripts.verify_dependencies import (
    get_package_version,
    verify_packages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_exists():
    """
    Verify that pyproject.toml exists.
    """

    pyproject = PROJECT_ROOT / "pyproject.toml"

    assert pyproject.is_file()


def test_pyproject_contains_project_metadata():
    """
    Verify required project metadata.
    """

    pyproject = PROJECT_ROOT / "pyproject.toml"

    with pyproject.open("rb") as file:
        configuration = tomllib.load(file)

    project = configuration["project"]

    assert project["name"] == "neojaundice"

    assert project["version"] == "0.1.0"

    assert project["requires-python"] == ">=3.11"


def test_required_dependency_groups_exist():
    """
    Verify that dependency groups are configured.
    """

    pyproject = PROJECT_ROOT / "pyproject.toml"

    with pyproject.open("rb") as file:
        configuration = tomllib.load(file)

    groups = configuration["project"]["optional-dependencies"]

    expected_groups = {
        "backend",
        "cv",
        "ml",
        "mlops",
        "monitoring",
        "dev",
    }

    assert expected_groups.issubset(groups.keys())


def test_missing_package_returns_none():
    """
    Verify missing package handling.
    """

    with patch(
        "scripts.verify_dependencies.importlib.metadata.version"
    ) as mock_version:

        from importlib.metadata import PackageNotFoundError

        mock_version.side_effect = PackageNotFoundError

        result = get_package_version("nonexistent-package")

        assert result is None


def test_verify_packages_returns_versions():
    """
    Verify package version collection.
    """

    with patch(
        "scripts.verify_dependencies.get_package_version"
    ) as mock_version:

        mock_version.return_value = "1.2.3"

        results = verify_packages(
            {
                "example-package": "example_module",
            }
        )

        assert results == {
            "example-package": "1.2.3",
        }
"""
Tests for NeoJaundice quality configuration.
"""

from scripts.verify_quality_config import (
    verify_ci_configuration,
    verify_coverage_configuration,
    verify_pre_commit_configuration,
    verify_pytest_configuration,
)


def test_pytest_configuration():
    """Verify pytest settings."""

    assert verify_pytest_configuration()


def test_coverage_configuration():
    """Verify coverage settings."""

    assert verify_coverage_configuration()


def test_pre_commit_configuration():
    """Verify pre-commit hooks."""

    assert verify_pre_commit_configuration()


def test_ci_configuration():
    """Verify GitHub Actions workflow."""

    assert verify_ci_configuration()

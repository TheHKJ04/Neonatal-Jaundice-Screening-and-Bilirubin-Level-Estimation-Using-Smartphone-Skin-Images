"""Tests for MLflow tracking URI configuration."""

import pytest


def resolve_tracking_uri(
    cli_uri: str | None,
    environment_uri: str | None,
    default_uri: str,
) -> str:
    """Resolve and validate the MLflow tracking URI."""
    uri = cli_uri or environment_uri or default_uri

    if uri.startswith("file:"):
        raise ValueError("Legacy filesystem tracking is unsupported.")

    return uri


def test_cli_uri_has_priority():
    result = resolve_tracking_uri(
        cli_uri="sqlite:////content/custom.db",
        environment_uri="sqlite:////content/env.db",
        default_uri="sqlite:////content/default.db",
    )

    assert result == "sqlite:////content/custom.db"


def test_environment_uri_used_when_cli_missing():
    result = resolve_tracking_uri(
        cli_uri=None,
        environment_uri="sqlite:////content/env.db",
        default_uri="sqlite:////content/default.db",
    )

    assert result == "sqlite:////content/env.db"


def test_default_uri_used_when_no_override():
    result = resolve_tracking_uri(
        cli_uri=None,
        environment_uri=None,
        default_uri="sqlite:////content/default.db",
    )

    assert result == "sqlite:////content/default.db"


def test_legacy_file_uri_rejected():
    with pytest.raises(ValueError):
        resolve_tracking_uri(
            cli_uri="file:///content/mlruns",
            environment_uri=None,
            default_uri="sqlite:////content/default.db",
        )

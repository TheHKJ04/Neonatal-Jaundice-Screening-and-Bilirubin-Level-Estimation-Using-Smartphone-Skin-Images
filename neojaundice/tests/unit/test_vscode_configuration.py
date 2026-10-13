# type: ignore
"""
Tests for NeoJaundice VS Code workspace configuration.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VSCODE_DIR = PROJECT_ROOT / ".vscode"


def load_json(filename: str) -> dict:
    """Load a VS Code JSON configuration file."""

    file_path = VSCODE_DIR / filename

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_vscode_directory_exists():
    """Verify the VS Code directory exists."""

    assert VSCODE_DIR.is_dir()


def test_settings_file_exists():
    """Verify settings.json exists."""

    assert (VSCODE_DIR / "settings.json").is_file()


def test_extensions_file_exists():
    """Verify extensions.json exists."""

    assert (VSCODE_DIR / "extensions.json").is_file()


def test_launch_file_exists():
    """Verify launch.json exists."""

    assert (VSCODE_DIR / "launch.json").is_file()


def test_python_testing_enabled():
    """Verify pytest integration is enabled."""

    settings = load_json("settings.json")

    assert settings["python.testing.pytestEnabled"] is True

    assert settings["python.testing.unittestEnabled"] is False


def test_format_on_save_enabled():
    """Verify automatic formatting is enabled."""

    settings = load_json("settings.json")

    assert settings["editor.formatOnSave"] is True


def test_required_extensions_recommended():
    """Verify essential extensions are recommended."""

    extensions = load_json("extensions.json")

    recommendations = set(extensions["recommendations"])

    required = {
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-python.debugpy",
    }

    assert required.issubset(recommendations)


def test_debug_configurations_exist():
    """Verify debugging configurations exist."""

    configuration = load_json("launch.json")

    configurations = configuration["configurations"]

    assert len(configurations) >= 4

    names = {item["name"] for item in configurations}

    assert "NeoJaundice: Run Unit Tests" in names


def test_python_interpreter_setting():
    """Verify Python interpreter configuration."""

    settings = load_json("settings.json")

    assert settings["python.defaultInterpreterPath"] == "python"

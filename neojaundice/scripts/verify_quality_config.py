"""
Verify NeoJaundice quality tooling configuration.
"""

import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

PRE_COMMIT_PATH = PROJECT_ROOT / ".pre-commit-config.yaml"

CI_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def load_pyproject() -> dict:
    """Load project configuration."""

    with PYPROJECT_PATH.open("rb") as file:
        return tomllib.load(file)


def load_yaml(path: Path) -> dict:
    """Load YAML configuration."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        result = yaml.safe_load(file)

    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return result


def verify_pytest_configuration() -> bool:
    """Verify pytest configuration exists."""

    configuration = load_pyproject()

    pytest_config = configuration["tool"]["pytest"]["ini_options"]

    assert "testpaths" in pytest_config

    assert "--strict-markers" in pytest_config["addopts"]

    return True


def verify_coverage_configuration() -> bool:
    """Verify coverage configuration exists."""

    configuration = load_pyproject()

    coverage_config = configuration["tool"]["coverage"]

    assert coverage_config["run"]["branch"] is True

    assert "backend" in coverage_config["run"]["source"]

    return True


def verify_pre_commit_configuration() -> bool:
    """Verify pre-commit hooks are configured."""

    configuration = load_yaml(PRE_COMMIT_PATH)

    repositories = configuration["repos"]

    hook_ids = {
        hook["id"] for repository in repositories for hook in repository["hooks"]
    }

    required_hooks = {
        "ruff",
        "ruff-format",
        "check-yaml",
        "check-toml",
        "check-json",
        "detect-private-key",
    }

    assert required_hooks.issubset(hook_ids)

    return True


def verify_ci_configuration() -> bool:
    """Verify CI workflow exists and defines jobs."""

    configuration = load_yaml(CI_PATH)

    assert "jobs" in configuration

    assert "quality" in configuration["jobs"]

    return True


def main() -> int:
    """Run quality configuration verification."""

    checks = (
        verify_pytest_configuration,
        verify_coverage_configuration,
        verify_pre_commit_configuration,
        verify_ci_configuration,
    )

    print("=" * 55)
    print("NeoJaundice Quality Configuration Verification")
    print("=" * 55)

    try:
        for check in checks:
            check()
            print(f"[OK] {check.__name__}")

    except (AssertionError, KeyError, OSError, ValueError) as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        return 1

    print("\n[PASSED] Quality configuration verified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

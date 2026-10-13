"""
Verify NeoJaundice development dependencies.

Usage:
    python scripts/verify_dependencies.py
"""

import importlib.metadata
import sys

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "pytest": "pytest",
    "ruff": "ruff",
    "mypy": "mypy",
}


def get_package_version(package_name: str) -> str | None:
    """
    Return an installed package version.

    Return None when the package is not installed.
    """

    try:
        return importlib.metadata.version(package_name)

    except importlib.metadata.PackageNotFoundError:
        return None


def verify_packages(
    required_packages: dict[str, str],
) -> dict[str, str | None]:
    """
    Check installed package versions.
    """

    results = {}

    for package_name in required_packages:
        results[package_name] = get_package_version(package_name)

    return results


def main() -> int:
    """
    Run dependency verification.
    """

    print("=" * 55)
    print("NeoJaundice Dependency Verification")
    print("=" * 55)

    print(f"\nPython executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    results = verify_packages(REQUIRED_PACKAGES)

    missing_packages = []

    print("\nInstalled packages:")

    for package_name, version in results.items():
        if version is None:
            print(f"[MISSING] {package_name}")
            missing_packages.append(package_name)

        else:
            print(f"[OK] {package_name}: {version}")

    if missing_packages:
        print("\n[FAILED] Missing dependencies.")

        return 1

    print("\n[PASSED] Required development dependencies found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Verify the NeoJaundice container environment.

This script does not start the API or load ML models.
"""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "pydantic",
    "pytest",
)


def check_python_version() -> bool:
    """Verify Python 3.13 is running."""

    return sys.version_info[:2] == (3, 13)


def check_packages() -> dict[str, str | None]:
    """Check required installed packages."""

    results = {}

    for package in REQUIRED_PACKAGES:
        try:
            results[package] = version(package)
        except PackageNotFoundError:
            results[package] = None

    return results


def main() -> int:
    """Run container smoke checks."""

    print("=" * 50)
    print("NeoJaundice Container Smoke Test")
    print("=" * 50)

    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.system()}")

    if not check_python_version():
        print("[FAILED] Expected Python 3.13")
        return 1

    packages = check_packages()

    for package, installed_version in packages.items():
        if installed_version is None:
            print(f"[MISSING] {package}")
            return 1

        print(f"[OK] {package}: {installed_version}")

    print("\n[PASSED] Container environment verified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Verify the NeoJaundice repository structure.

This script checks that required project directories
and Python package initializers exist.

Usage:
    python scripts/verify_structure.py
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_DIRECTORIES = (
    ".github/workflows",
    "backend/app",
    "backend/app/api",
    "backend/app/core",
    "backend/app/models",
    "backend/app/repositories",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/utils",
    "backend/tests",
    "configs/data",
    "configs/deployment",
    "configs/inference",
    "configs/model",
    "configs/training",
    "data/external",
    "data/interim",
    "data/processed",
    "data/raw",
    "docker",
    "docs",
    "experiments",
    "frontend/src/components",
    "frontend/src/hooks",
    "frontend/src/pages",
    "frontend/src/services",
    "frontend/src/types",
    "logs",
    "ml",
    "ml/datasets",
    "ml/evaluation",
    "ml/explainability",
    "ml/features",
    "ml/inference",
    "ml/preprocessing",
    "ml/roi",
    "ml/training",
    "mlflow",
    "models/baseline",
    "models/efficientnet",
    "models/multimodal",
    "models/onnx",
    "models/registry",
    "notebooks",
    "scripts",
    "tests/api",
    "tests/integration",
    "tests/model",
    "tests/unit",
)


PYTHON_PACKAGES = (
    "backend/app",
    "backend/app/api",
    "backend/app/core",
    "backend/app/models",
    "backend/app/repositories",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/utils",
    "ml",
    "ml/datasets",
    "ml/evaluation",
    "ml/explainability",
    "ml/features",
    "ml/inference",
    "ml/preprocessing",
    "ml/roi",
    "ml/training",
)


def verify_directories(root: Path) -> list[str]:
    """Return missing required directories."""

    missing = []

    for directory in REQUIRED_DIRECTORIES:
        path = root / directory

        if not path.is_dir():
            missing.append(directory)

    return missing


def verify_python_packages(root: Path) -> list[str]:
    """Return packages missing their initializer."""

    missing = []

    for package in PYTHON_PACKAGES:
        initializer = root / package / "__init__.py"

        if not initializer.is_file():
            missing.append(str(initializer.relative_to(root)))

    return missing


def verify_structure(root: Path) -> dict[str, list[str]]:
    """Validate the repository structure."""

    return {
        "missing_directories": verify_directories(root),
        "missing_initializers": verify_python_packages(root),
    }


def main() -> int:
    """Run verification and return an exit code."""

    print("=" * 50)
    print("NeoJaundice Project Structure Verification")
    print("=" * 50)

    results = verify_structure(PROJECT_ROOT)

    missing_directories = results["missing_directories"]
    missing_initializers = results["missing_initializers"]

    if missing_directories:
        print("\nMissing directories:")

        for directory in missing_directories:
            print(f"  [MISSING] {directory}")

    if missing_initializers:
        print("\nMissing Python package initializers:")

        for initializer in missing_initializers:
            print(f"  [MISSING] {initializer}")

    if missing_directories or missing_initializers:
        print("\n[FAILED] Repository structure is incomplete.")
        return 1

    print(f"\nVerified directories: {len(REQUIRED_DIRECTORIES)}")
    print(f"Verified Python packages: {len(PYTHON_PACKAGES)}")

    print("\n[PASSED] Repository structure is valid.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
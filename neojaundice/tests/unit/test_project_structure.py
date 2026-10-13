"""
Unit tests for NeoJaundice project structure.
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_structure import (  # noqa: E402
    PYTHON_PACKAGES,
    REQUIRED_DIRECTORIES,
    verify_directories,
    verify_python_packages,
    verify_structure,
)


class TestProjectStructure(unittest.TestCase):
    def test_actual_project_structure(self):
        """Verify the real project structure."""

        result = verify_structure(PROJECT_ROOT)

        self.assertEqual(
            result["missing_directories"],
            [],
        )

        self.assertEqual(
            result["missing_initializers"],
            [],
        )

    def test_missing_directories_detected(self):
        """Verify missing directories are detected."""

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            missing = verify_directories(root)

            self.assertEqual(
                len(missing),
                len(REQUIRED_DIRECTORIES),
            )

    def test_missing_initializers_detected(self):
        """Verify missing package initializers are detected."""

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            missing = verify_python_packages(root)

            self.assertEqual(
                len(missing),
                len(PYTHON_PACKAGES),
            )

    def test_valid_temporary_structure(self):
        """Verify a complete structure passes validation."""

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            for directory in REQUIRED_DIRECTORIES:
                (root / directory).mkdir(
                    parents=True,
                    exist_ok=True,
                )

            for package in PYTHON_PACKAGES:
                initializer = root / package / "__init__.py"

                initializer.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                initializer.touch()

            result = verify_structure(root)

            self.assertEqual(
                result["missing_directories"],
                [],
            )

            self.assertEqual(
                result["missing_initializers"],
                [],
            )


if __name__ == "__main__":
    unittest.main()

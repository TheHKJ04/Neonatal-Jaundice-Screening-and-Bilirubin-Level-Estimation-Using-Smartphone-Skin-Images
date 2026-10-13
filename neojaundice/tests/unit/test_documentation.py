# type: ignore
"""
Tests for NeoJaundice project documentation.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCUMENTS = (
    "README.md",
    "docs/index.md",
    "docs/setup.md",
    "docs/architecture.md",
    "docs/api.md",
    "docs/ml_pipeline.md",
    "docs/research_methodology.md",
    "docs/deployment.md",
    "docs/security.md",
    "docs/docker.md",
    "docs/testing.md",
    "docs/configuration.md",
    "docs/troubleshooting.md",
)

MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def test_required_documentation_exists():
    """Verify required documentation files exist."""

    missing = [
        relative_path
        for relative_path in REQUIRED_DOCUMENTS
        if not (PROJECT_ROOT / relative_path).is_file()
    ]

    assert not missing, f"Missing documentation files: {missing}"


def test_documentation_files_not_empty():
    """Verify documentation files contain content."""

    for relative_path in REQUIRED_DOCUMENTS:
        document = PROJECT_ROOT / relative_path

        content = document.read_text(encoding="utf-8")

        assert content.strip(), f"Documentation file is empty: {relative_path}"


def test_readme_contains_project_name():
    """Verify README identifies the project."""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "NeoJaundice" in readme


def test_readme_contains_medical_disclaimer():
    """Verify README identifies clinical limitations."""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "not a clinically validated" in readme.lower()


def test_documentation_internal_links():
    """Verify local Markdown links resolve."""

    broken_links = []

    for relative_path in REQUIRED_DOCUMENTS:
        document = PROJECT_ROOT / relative_path

        content = document.read_text(encoding="utf-8")

        links = MARKDOWN_LINK_PATTERN.findall(content)

        for link in links:
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue

            link_path = link.split("#", maxsplit=1)[0]

            if not link_path:
                continue

            target = (document.parent / link_path).resolve()

            if not target.is_relative_to(PROJECT_ROOT.resolve()):
                broken_links.append(f"{relative_path}: link escapes repository: {link}")
                continue

            if not target.exists():
                broken_links.append(f"{relative_path}: {link}")

    assert not broken_links, "Broken internal documentation links:\n" + "\n".join(
        broken_links
    )

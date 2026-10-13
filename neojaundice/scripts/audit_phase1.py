"""
NeoJaundice Phase 1 Engineering Audit.

Runs non-destructive checks for the development foundation.

Usage:
    python scripts/audit_phase1.py

The audit does not:
- Modify source files
- Delete Docker volumes
- Start production services
- Upload project data
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIR = PROJECT_ROOT / "docs" / "audits"

JSON_REPORT = AUDIT_DIR / "phase1_audit.json"

MARKDOWN_REPORT = AUDIT_DIR / "phase1_audit.md"

COMMAND_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class AuditResult:
    """Result of a single audit check."""

    name: str
    status: str
    details: str


def run_command(
    name: str,
    command: Sequence[str],
    *,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
) -> AuditResult:
    """Run a command and capture its verification status."""

    try:
        completed = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    except FileNotFoundError:
        return AuditResult(
            name=name,
            status="FAIL",
            details="Required executable was not found.",
        )

    except subprocess.TimeoutExpired:
        return AuditResult(
            name=name,
            status="FAIL",
            details=f"Command exceeded {timeout} seconds.",
        )

    if completed.returncode == 0:
        return AuditResult(
            name=name,
            status="PASS",
            details="Command completed successfully.",
        )

    return AuditResult(
        name=name,
        status="FAIL",
        details=(
            f"Command exited with code {completed.returncode}. "
            "Inspect the command output separately."
        ),
    )


def check_python_version() -> AuditResult:
    """Verify the expected Python major/minor version."""

    actual = sys.version_info[:2]

    if actual == (3, 13):
        return AuditResult(
            name="Python 3.13",
            status="PASS",
            details="Python 3.13 detected.",
        )

    return AuditResult(
        name="Python 3.13",
        status="FAIL",
        details=f"Expected Python 3.13, found {actual}.",
    )


def check_git_ignored_files() -> AuditResult:
    """Check whether Git tracks files matching ignore rules."""

    try:
        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "-ci",
                "--exclude-standard",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return AuditResult(
            name="Git ignored-file audit",
            status="FAIL",
            details="Git audit could not be executed.",
        )

    if completed.returncode != 0:
        return AuditResult(
            name="Git ignored-file audit",
            status="FAIL",
            details="Git returned an error.",
        )

    tracked_ignored_files = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]

    if tracked_ignored_files:
        return AuditResult(
            name="Git ignored-file audit",
            status="FAIL",
            details=(f"{len(tracked_ignored_files)} tracked files match ignore rules."),
        )

    return AuditResult(
        name="Git ignored-file audit",
        status="PASS",
        details="No tracked files match active ignore rules.",
    )


def check_required_files() -> AuditResult:
    """Verify essential Phase 1 project files exist."""

    required_files = (
        "pyproject.toml",
        ".gitignore",
        ".dockerignore",
        ".env.example",
        ".pre-commit-config.yaml",
        "compose.yaml",
        "docker/Dockerfile.dev",
        ".github/workflows/ci.yml",
        "README.md",
        "docs/setup.md",
        "docs/architecture.md",
        "docs/testing.md",
        "docs/docker.md",
        "docs/security.md",
        "docs/research_methodology.md",
    )

    missing = [path for path in required_files if not (PROJECT_ROOT / path).is_file()]

    if missing:
        return AuditResult(
            name="Required project files",
            status="FAIL",
            details=f"{len(missing)} required files are missing.",
        )

    return AuditResult(
        name="Required project files",
        status="PASS",
        details=f"{len(required_files)} required files found.",
    )


def get_audit_commands() -> tuple[tuple[str, list[str]], ...]:
    """Return the Phase 1 verification commands."""

    python = sys.executable

    return (
        (
            "Dependency verification",
            [python, "scripts/verify_dependencies.py"],
        ),
        (
            "Dependency consistency",
            [python, "-m", "pip", "check"],
        ),
        (
            "Project structure",
            [python, "scripts/verify_structure.py"],
        ),
        (
            "Application configuration",
            [python, "scripts/verify_config.py"],
        ),
        (
            "Quality configuration",
            [python, "scripts/verify_quality_config.py"],
        ),
        (
            "Ruff linting",
            [
                python,
                "-m",
                "ruff",
                "check",
                "scripts",
                "backend",
                "ml",
                "tests",
            ],
        ),
        (
            "Ruff formatting",
            [
                python,
                "-m",
                "ruff",
                "format",
                "--check",
                "scripts",
                "backend",
                "ml",
                "tests",
            ],
        ),
        (
            "Unit tests",
            [
                python,
                "-m",
                "pytest",
                "tests/unit",
                "-q",
            ],
        ),
        (
            "Docker Compose configuration",
            [
                "docker",
                "compose",
                "config",
                "--quiet",
            ],
        ),
    )


def run_audit() -> list[AuditResult]:
    """Execute all configured audit checks."""

    results = [
        check_python_version(),
        check_required_files(),
        check_git_ignored_files(),
    ]

    for name, command in get_audit_commands():
        print(f"Checking: {name}", flush=True)

        results.append(
            run_command(
                name=name,
                command=command,
            )
        )

    return results


def generate_json_report(
    results: list[AuditResult],
) -> None:
    """Write the machine-readable audit report."""

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "project": "NeoJaundice",
        "phase": "Phase 1",
        "audit_type": "Engineering Foundation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(result) for result in results],
        "summary": {
            "total": len(results),
            "passed": sum(result.status == "PASS" for result in results),
            "failed": sum(result.status == "FAIL" for result in results),
        },
    }

    JSON_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


def generate_markdown_report(
    results: list[AuditResult],
) -> None:
    """Write a human-readable audit report."""

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    passed = sum(result.status == "PASS" for result in results)

    failed = len(results) - passed

    lines = [
        "# NeoJaundice — Phase 1 Engineering Audit",
        "",
        "## Summary",
        "",
        f"- Total checks: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        "",
        "## Verification Results",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]

    for result in results:
        safe_details = result.details.replace("|", "\\|")

        lines.append(f"| {result.name} | {result.status} | {safe_details} |")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This audit verifies selected local engineering checks.",
            "- It does not verify GitHub Actions execution.",
            "- It does not establish clinical model validity.",
            "- It does not constitute a security penetration test.",
            "- It does not prove exact dependency reproducibility.",
            "",
        ]
    )

    MARKDOWN_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    """Execute the Phase 1 audit."""

    print("=" * 60)
    print("NeoJaundice — Phase 1 Engineering Audit")
    print("=" * 60)

    results = run_audit()

    generate_json_report(results)
    generate_markdown_report(results)

    print()

    for result in results:
        print(f"[{result.status}] {result.name}: {result.details}")

    passed = sum(result.status == "PASS" for result in results)

    failed = len(results) - passed

    print()
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    print()
    print("Reports generated:")
    print(f"  {JSON_REPORT}")
    print(f"  {MARKDOWN_REPORT}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

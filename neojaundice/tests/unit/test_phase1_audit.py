"""
Unit tests for the Phase 1 engineering audit.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.audit_phase1 import (
    AuditResult,
    check_python_version,
    generate_json_report,
    generate_markdown_report,
    run_command,
)


def test_audit_result_creation():
    """Verify audit result structure."""

    result = AuditResult(
        name="Example",
        status="PASS",
        details="Verified",
    )

    assert result.name == "Example"
    assert result.status == "PASS"
    assert result.details == "Verified"


def test_python_version_check():
    """Verify Python 3.13 detection."""

    result = check_python_version()

    assert result.status == "PASS"


def test_successful_command():
    """Verify successful command handling."""

    result = run_command(
        name="Python execution",
        command=[
            "python",
            "--version",
        ],
    )

    assert result.status == "PASS"


def test_failed_command():
    """Verify nonzero exit codes are detected."""

    result = run_command(
        name="Intentional failure",
        command=[
            "python",
            "-c",
            "raise SystemExit(1)",
        ],
    )

    assert result.status == "FAIL"


def test_missing_executable():
    """Verify missing executable handling."""

    with patch(
        "scripts.audit_phase1.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        result = run_command(
            name="Missing tool",
            command=["missing-tool"],
        )

    assert result.status == "FAIL"


def test_command_timeout():
    """Verify command timeout handling."""

    with patch(
        "scripts.audit_phase1.subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            cmd=["test"],
            timeout=1,
        ),
    ):
        result = run_command(
            name="Timeout test",
            command=["test"],
        )

    assert result.status == "FAIL"


def test_json_report(tmp_path: Path):
    """Verify JSON report generation."""

    results = [
        AuditResult(
            name="Check A",
            status="PASS",
            details="Passed",
        ),
        AuditResult(
            name="Check B",
            status="FAIL",
            details="Failed",
        ),
    ]

    report_path = tmp_path / "audit.json"

    with (
        patch(
            "scripts.audit_phase1.AUDIT_DIR",
            tmp_path,
        ),
        patch(
            "scripts.audit_phase1.JSON_REPORT",
            report_path,
        ),
    ):
        generate_json_report(results)

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["summary"]["total"] == 2
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 1


def test_markdown_report(tmp_path: Path):
    """Verify Markdown report generation."""

    results = [
        AuditResult(
            name="Example check",
            status="PASS",
            details="Verified",
        )
    ]

    report_path = tmp_path / "audit.md"

    with (
        patch(
            "scripts.audit_phase1.AUDIT_DIR",
            tmp_path,
        ),
        patch(
            "scripts.audit_phase1.MARKDOWN_REPORT",
            report_path,
        ),
    ):
        generate_markdown_report(results)

    content = report_path.read_text(encoding="utf-8")

    assert "Phase 1 Engineering Audit" in content
    assert "Example check" in content
    assert "PASS" in content

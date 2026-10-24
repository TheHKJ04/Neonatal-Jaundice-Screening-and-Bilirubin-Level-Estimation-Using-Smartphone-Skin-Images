"""Static checks for the containerization contract; no Docker daemon required."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_runs_non_root() -> None:
    content = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert "USER app" in content
    assert '"backend.app.main:app"' in content
    assert "python:3.12-slim" in content


def test_checkpoints_not_baked_into_image() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "*.pt" in ignored
    assert "*.pth" in ignored
    compose = (ROOT / "docker-compose.api.yml").read_text(encoding="utf-8")
    assert "read_only: true" in compose
    assert "/app/checkpoints" in compose


def test_api_port_is_local_only() -> None:
    compose = (ROOT / "docker-compose.api.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:8001:8000" in compose

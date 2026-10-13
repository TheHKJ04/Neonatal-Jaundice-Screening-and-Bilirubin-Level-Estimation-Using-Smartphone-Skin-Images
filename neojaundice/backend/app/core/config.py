"""
Application configuration for NeoJaundice.

Environment variables are loaded through Pydantic Settings.

The module defines configuration only. It does not connect
to databases or load machine learning models.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "NeoJaundice"

    app_env: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    debug: bool = False

    api_host: str = "127.0.0.1"

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    db_host: str = "localhost"

    db_port: int = Field(
        default=5432,
        ge=1,
        le=65535,
    )

    db_name: str = "neojaundice"

    db_user: str = "neojaundice"

    db_password: SecretStr = SecretStr("")

    secret_key: SecretStr = SecretStr("")

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    model_path: Path = Path("models/registry")

    device: Literal["cpu", "cuda"] = "cpu"


@lru_cache
def get_settings() -> Settings:
    """
    Return cached application settings.

    Caching avoids repeatedly parsing environment files.
    """

    return Settings()

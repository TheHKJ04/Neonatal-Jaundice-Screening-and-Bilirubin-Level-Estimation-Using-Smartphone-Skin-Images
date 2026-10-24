"""Environment-driven backend settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/app -> backend -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEOJAUNDICE_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "NeoJaundice"

    app_env: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    regression_checkpoint: Path = Field(
        default=REPOSITORY_ROOT / "artifacts/transferred/regression_best_model.pt"
    )

    classification_checkpoint: Path = Field(
        default=REPOSITORY_ROOT / "artifacts/transferred/classification_best_model.pt"
    )

    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=50 * 1024 * 1024,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

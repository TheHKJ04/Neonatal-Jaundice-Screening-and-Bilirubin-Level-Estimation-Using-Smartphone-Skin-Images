"""Public API response models."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str


class ModelStatus(BaseModel):
    available: bool
    loaded: bool = False


class ModelsResponse(BaseModel):
    regression: ModelStatus
    classification: ModelStatus
    research_only: bool = True

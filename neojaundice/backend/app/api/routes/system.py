"""Health and model inventory routes."""

from fastapi import APIRouter, Request

from backend.app.schemas.system import HealthResponse, ModelsResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(service=request.app.state.settings.app_name)


@router.get("/models/status", response_model=ModelsResponse)
def model_status(request: Request) -> ModelsResponse:
    return request.app.state.model_registry.status()

"""FastAPI application factory."""

from fastapi import FastAPI

from backend.app.api.routes.metrics import router as metrics_router
from backend.app.api.routes.predictions import (
    router as predictions_router,
)
from backend.app.api.routes.system import router as system_router
from backend.app.core.config import Settings, get_settings
from backend.app.monitoring.logging_config import configure_logging
from backend.app.monitoring.middleware import ObservabilityMiddleware
from backend.app.services.model_registry import ModelRegistry


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct the API application and register its dependencies."""

    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        description=(
            "NeoJaundice research API for image-based model inference. "
            "This prototype is not a medical device or clinical diagnostic service."
        ),
    )

    app.state.settings = settings

    app.state.model_registry = ModelRegistry(
        regression_checkpoint=settings.regression_checkpoint,
        classification_checkpoint=settings.classification_checkpoint,
    )

    configure_logging()
    app.add_middleware(ObservabilityMiddleware)
    app.include_router(metrics_router)
    app.include_router(
        system_router,
        prefix="/api/v1",
    )

    app.include_router(
        predictions_router,
        prefix="/api/v1",
    )

    return app


app = create_app()

"""Validated response schema for NeoJaundice research predictions."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    """Numerical outputs from the two research models."""

    model_config = ConfigDict(extra="forbid")

    regression_prediction: float = Field(
        description=(
            "Raw numerical output of the regression model. "
            "Its clinical units must be verified against dataset documentation."
        )
    )

    classification_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Sigmoid-transformed score from the binary classification model. "
            "This is not a clinically calibrated probability."
        ),
    )

    classification_prediction: int = Field(
        ge=0,
        le=1,
        description=(
            "Binary model output using a 0.5 probability threshold. "
            "It is not a diagnosis or treatment recommendation."
        ),
    )

    research_only: bool = True

    message: str = (
        "Research prototype only. Not intended for clinical diagnosis "
        "or treatment decisions."
    )

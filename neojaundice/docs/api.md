# API contract and verification

## Current frontend dependency

The frontend calls `POST /api/v1/predict` with multipart form data:

- field: `file`
- accepted image types in the UI: JPEG, PNG, WebP
- UI maximum size: 10 MiB

The current frontend expects the response to include:

```json
{
  "regression_prediction": 10.25,
  "classification_prediction": 1,
  "classification_probability": 0.82,
  "research_only": true,
  "message": "Research output only."
}
```

This JSON is an illustrative contract example, not a measured prediction or an assertion about a specific running model. Confirm it against the current FastAPI Pydantic schema.

## Required validation

- Check that regression output is finite and the unit is documented as mg/dL.
- Confirm whether `classification_probability` is the probability of class 1, the predicted class, or another score.
- Confirm class-label mapping from source dataset and training code.
- Return clear HTTP errors for invalid file type, invalid image content, oversized files, and model-not-ready states.
- Keep a `research_only` flag and message until clinical validation and governance are completed.

## Local endpoints

- Health: `GET /api/v1/health`
- Prediction: `POST /api/v1/predict`
- Interactive docs: `/docs` on the API host port

Do not document routes as implemented until confirmed in the backend route definitions and a runtime request.

# NeoJaundice API Documentation

## Status

**Planned — API endpoints are not implemented yet.**

## Proposed Framework

FastAPI

## Proposed API Responsibilities

- Validate incoming requests
- Accept image uploads
- Validate image formats
- Coordinate inference
- Return structured prediction responses
- Record permitted metadata
- Expose health and readiness endpoints

## Proposed API Versioning

```text
/api/v1/
```

## Candidate Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service health |
| GET | `/ready` | Service readiness |
| POST | `/api/v1/predictions` | Submit an image for inference |
| GET | `/api/v1/models/current` | Retrieve model metadata |

These endpoints are proposed and subject to revision.

## Proposed Prediction Response

```json
{
  "prediction_id": "example-id",
  "model_version": "research-model-version",
  "quality_status": "accepted",
  "screening_output": null,
  "bilirubin_estimate_mg_dl": null,
  "status": "not_implemented"
}
```

This is a documentation example, not an actual model response.

## Validation Requirements

The API should validate:

- File type
- File size
- Image dimensions
- Decoding success
- Image quality
- Request metadata
- Authentication and authorization

## Error Handling

Use structured errors with appropriate HTTP status codes.

Do not expose stack traces or internal credentials.

## API Documentation Generation

When FastAPI is implemented, OpenAPI documentation
will be generated from the application schemas.

## Security Requirements

Image uploads must be handled as untrusted input.

The API must not log identifiable patient information
or raw medical images by default.

## Medical Safety

Inference responses must clearly communicate model
limitations and must not imply clinical authorization.

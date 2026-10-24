# Backend setup — Module 1

## Purpose
Provide a testable API foundation without loading model checkpoints or exposing patient data.

## Installation (Windows PowerShell)
Run from the repository root (`D:\CV Project\neojaundice`) with Python 3.12 activated:

```powershell
python -m pip install -r backend/requirements.txt
python -m pytest tests/backend -q
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/docs` for Swagger UI.

## Endpoints
- `GET /api/v1/health`: process is responding.
- `GET /api/v1/models/status`: checks whether checkpoint *files* exist. `loaded` is always false until an inference adapter is implemented. Do not confuse file presence with a valid or usable model.

## Image Inference API

### Endpoint

POST /api/v1/predict

Content type: multipart/form-data

Form field:
- file: JPEG, PNG, or WebP image

Maximum upload size:
- 10 MiB by default
- Configurable through NEOJAUNDICE_MAX_UPLOAD_BYTES

### Response

- bilirubin_regression_output:
  Numerical output on the model's training-target scale.
  Its unit and measurement type must be verified before interpretation.

- treatment_label_probability:
  Sigmoid-transformed classification output for the dataset's treatment_label.

- predicted_treatment_label:
  Binary prediction using a 0.5 probability threshold.

- research_only:
  Always true.

- warning:
  Explains that the output is not a diagnosis or treatment recommendation.

### Error responses

- 400: Empty or invalid image.
- 413: Upload exceeds the configured limit.
- 415: Unsupported image format.
- 503: Required model is unavailable or cannot be loaded.
- 500: Unexpected inference failure.

### Important limitations

This API is a research prototype, not a clinical diagnostic service.
Model outputs must not be used independently to diagnose neonatal jaundice
or make treatment decisions.

Clinical validation, target-unit verification, calibration assessment,
and external validation remain separate research requirements.

## Architecture
`api/routes` -> `services` -> checkpoint metadata. `core` stores environment configuration and `schemas` define response contracts.

## Configuration
The default checkpoint paths are computed from the repository root. To override them, set `NEOJAUNDICE_REGRESSION_CHECKPOINT` and `NEOJAUNDICE_CLASSIFICATION_CHECKPOINT` in the environment, or copy `backend/.env.example` to `.env` in the working directory.

## Security and clinical boundaries
- No patient images are uploaded or stored in this module.
- No checkpoint is deserialized in the model-status endpoint. Only use verified, trusted checkpoints when adding PyTorch loading.
- This API is research-only, not a validated clinical device. Do not use its future estimates for treatment decisions.
- Use loopback binding for development. Add authentication, request limits, TLS, audit logging, and privacy review before external exposure.

## Troubleshooting
- `ModuleNotFoundError: backend`: start the server/tests from the repository root.
- Missing `pydantic_settings`: install `backend/requirements.txt` in the active environment.
- Checkpoints unavailable: inspect file paths and confirm `.pt` files exist; status does not load them.
- Port 8000 in use: choose `--port 8001`.

## Next module
Create a checkpoint-loading adapter with architecture verification, input transforms matching training, inference contracts, and tests using synthetic images only.

## Real-Model Integration Testing and Inference Benchmarking

### Integration testing

The integration test uses the configured regression and classification
checkpoints and the project's existing inference preprocessing.

A synthetic image is used to validate checkpoint loading, inference,
output types, finite regression output, classification score bounds,
and consistency across repeated predictions.

The test is skipped if the required checkpoint files are unavailable.
A skipped test must not be reported as successful real-model validation.

### Performance benchmarking

The benchmark script measures:

- Cold-start latency, including lazy checkpoint loading.
- Mean warm inference latency.
- Median warm inference latency.
- Approximate 95th-percentile warm inference latency.
- Minimum and maximum warm inference latency.

Results are written to:

`reports/performance/inference_benchmark.json`

### Reproducibility

For meaningful comparisons, record the checkpoint versions, Python and
PyTorch versions, hardware, operating system, image preprocessing,
and number of warm repetitions.

Use the same checkpoint, input image, and machine when comparing changes.

### Limitations

The initial benchmark runs locally on CPU. It does not measure concurrent
HTTP requests, deployment performance, or clinical validity.

Synthetic-image integration tests verify software behavior, not the
accuracy of neonatal bilirubin estimation.

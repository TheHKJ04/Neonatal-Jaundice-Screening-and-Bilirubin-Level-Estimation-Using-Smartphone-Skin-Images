# System architecture

## Runtime path

```text
Browser
  |
  v
React + TypeScript + Vite build (static files)
  |
  v
Nginx frontend container :8080 -> :80
  |  /api/* reverse proxy
  v
FastAPI service `api:8000` (host-published on 127.0.0.1:8001)
  |
  v
Model loading / image validation / inference
  |
  +--> regression estimate (mg/dL)
  +--> binary classifier output + reported probability
```

## Boundaries

- **Frontend:** file selection/validation, preview, theme, presentation, API call, and safety messaging. It must not invent clinical thresholds.
- **Nginx:** serves the compiled single-page app, handles SPA fallback, and proxies `/api/` to FastAPI.
- **FastAPI:** validates requests, loads the configured model checkpoints, performs inference, and returns a typed response.
- **ML package/scripts:** own dataset semantics, preprocessing, training, evaluation, and reproducibility.
- **PostgreSQL, MLflow, DVC, monitoring:** only describe as active runtime dependencies when the relevant services/configuration are present and verified. Do not imply this frontend patch adds them.

## Clean Architecture principle

Keep UI concerns in React, HTTP/inference concerns in the API layer, and model/data concerns in the ML domain. Avoid embedding model thresholds or training-label assumptions in the frontend.

## Current known limitation

The classifier was trained against `treatment_label`; its clinical meaning is recorded as unverified in the classification training/evaluation artifacts. The current frontend maps numeric classes to readable **provisional** labels only. This must be resolved from the source dataset documentation before clinical interpretation.

# Project blueprint (scope and status)

## Goal
Research neonatal jaundice screening and bilirubin estimation from smartphone skin images, with a React frontend, FastAPI inference API, reproducible ML workflows, and containerized deployment.

## Current frontend/backend workflow
See [`README.md`](README.md) for the quick start and [`docs/architecture.md`](docs/architecture.md) for component boundaries.

## Intended roadmap
1. Dataset validation and patient-level split audit.
2. Preprocessing/ROI reproducibility and baselines.
3. Regression and classification evaluation on frozen held-out test data.
4. API schema and checkpoint provenance verification.
5. Frontend and API containerization.
6. MLflow/DVC/CI/monitoring integration and verification.
7. Research report, model limitations, and deployment documentation.

Roadmap items are not evidence of implementation. See [`docs/README.md`](docs/README.md) for the canonical docs and status definitions.

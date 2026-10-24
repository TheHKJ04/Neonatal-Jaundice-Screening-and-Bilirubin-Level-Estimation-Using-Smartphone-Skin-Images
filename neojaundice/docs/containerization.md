# Module 5A — Containerized FastAPI

## Scope
This stage containerizes FastAPI separately from the working Module 4 monitoring stack. It does **not** change the existing Prometheus/Grafana containers or claim production/clinical readiness.

## Prerequisites
Docker Desktop, Python 3.12 project, installable `pyproject.toml`, trained checkpoint files. Verify exact checkpoint environment variable names in `backend/app/core/config.py` before running real inference.

## Setup (PowerShell, repository root)
1. Copy `.env.docker.example` to `.env.docker` and update `NEOJAUNDICE_CHECKPOINT_DIR` to an existing local directory containing checkpoints.
2. Ensure `.env.docker` is in `.gitignore`.
3. `docker compose --env-file .env.docker -f docker-compose.api.yml config`
4. `docker compose --env-file .env.docker -f docker-compose.api.yml build`
5. `docker compose --env-file .env.docker -f docker-compose.api.yml up -d`
6. `docker compose --env-file .env.docker -f docker-compose.api.yml ps`
7. `Invoke-RestMethod http://127.0.0.1:8001/api/v1/health`
8. `Invoke-WebRequest http://127.0.0.1:8001/metrics`
9. `docker compose --env-file .env.docker -f docker-compose.api.yml logs --tail=80 api`

The original host FastAPI can remain on port 8000; the Docker API uses localhost:8001. Prometheus remains pointed at the host API until Module 5B networking integration.

## Model path integration
The Compose file mounts checkpoint directory read-only at `/app/checkpoints`. To enable inference, set the *actual* checkpoint environment variable names from `backend/app/core/config.py` to container paths. Do not guess or hardcode checkpoint names. Test `/api/v1/models/status` (or your configured model status endpoint) before `/api/v1/predict`.

## Limitations and next phase
The Dockerfile assumes `pip install .` installs all required runtime packages and that `backend` and `ml` are importable. If build fails, inspect the actual pyproject dependencies and packaging rules before changing the image. Module 5B will move Prometheus scraping to the container network after Module 5A passes tests. Record Docker image digest, Git commit, dataset revision, model hashes, and MLflow run IDs for reproducibility. Never log patient images or identifiers.

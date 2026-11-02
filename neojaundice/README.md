# NeoJaundice

**Research project:** neonatal jaundice screening and bilirubin estimation using smartphone skin images.

> **Safety status:** research prototype only. The image-derived estimate and classifier output are not a clinically validated and must not be used to diagnose a newborn, rule out jaundice, delay medical care, or make treatment decisions.

## Current components

- `frontend/`: React + TypeScript + Vite, served in production by Nginx.
- `Dockerfile.api` and `docker-compose.api.yml`: FastAPI research inference service.
- `ml/` and `scripts/`: training, evaluation, preprocessing, and experiment workflows.
- `docs/`: canonical project, setup, API, ML, deployment, troubleshooting, and research documentation.

## Quick start (Windows PowerShell)

### 1. Configure backend model checkpoints

Create/update `.env.docker` as required by `docker-compose.api.yml`. Set `NEOJAUNDICE_CHECKPOINT_DIR` to the directory containing the checkpoint files expected by the backend. Do not commit `.env.docker` or private data.

### 2. Build and start API + frontend

```powershell
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml up --build -d
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml ps
```

Open the frontend at `http://localhost:8080`. The API is proxied by Nginx to the Compose service `api:8000`. API health is available at `http://127.0.0.1:8001/api/v1/health` and Swagger at `http://127.0.0.1:8001/docs`.

Stop services:

```powershell
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml down
```

### 3. Frontend development mode

```powershell
cd frontend
npm ci
npm run dev
```

Development mode uses the Vite proxy configuration. Start the API separately as described in `docs/setup.md`.

## Validation

```powershell
cd frontend
npm run test:run
npm run build
npm run lint
```

Run backend tests from the repository root using the project's configured Python environment; see `docs/setup.md`.

## Classification wording

The frontend renders binary outputs as **“Jaundice signal indicated (provisional)”** for class 1 and **“No jaundice signal indicated (provisional)”** for class 0. This is a UI-only provisional mapping. The current classifier target is `treatment_label`, and the source label semantics have not yet been verified. Confirm the dataset definition and backend probability semantics before interpreting these labels. A class-0 result never rules out jaundice.

## Documentation

Start at [`docs/README.md`](docs/README.md). Root-level legacy documents are kept short and point to the canonical docs rather than repeating full specifications.

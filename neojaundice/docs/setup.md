# Setup and validation

## Prerequisites

- Windows 10/11 with PowerShell, Git, Docker Desktop, and Node.js 22+ for local frontend development.
- Python environment compatible with the root `pyproject.toml` for ML/backend scripts.
- `frontend/package-lock.json` must be committed so Docker's `npm ci` is reproducible.
- Model checkpoint directory configured in `.env.docker`.

## Start API + frontend in Docker

From repository root:

```powershell
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml config
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml up --build -d
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml ps
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml logs --tail=100 api frontend
```

Browse `http://localhost:8080`. Check `http://127.0.0.1:8001/api/v1/health` and `http://127.0.0.1:8001/docs`.

Stop:

```powershell
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml down
```

## Frontend development

```powershell
cd frontend
npm ci
npm run dev
npm run test:run
npm run build
npm run lint
```

The current Vite proxy expects the API at `http://127.0.0.1:8001`; check `vite.config.ts` before changing this.

## Reproducibility

- Do not commit `.env`, `.env.docker`, private datasets, or patient-identifying images.
- Record the Python version and lockfile/dependency state when documenting test results.
- Keep test outputs and evaluation reports separate from source code; avoid committing caches or generated build directories.
- A clean Docker build and successful health check do not by themselves prove the expected model checkpoint is loaded. Verify model status and inference separately.

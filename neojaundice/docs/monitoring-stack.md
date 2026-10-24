# Module 4: Monitoring Stack

## Purpose
Local, research-only observability for the NeoJaundice FastAPI service. Monitoring infrastructure does not establish clinical validity.

## Architecture (text)
1. FastAPI records HTTP and model inference metrics.
2. Prometheus scrapes `http://host.docker.internal:8000/metrics` every 15 seconds.
3. Grafana reads Prometheus using `http://prometheus:9090`.
4. Dashboards display request rate, HTTP errors, latency, inference errors and route traffic.

## Requirements
- Windows Docker Desktop with Linux containers enabled.
- FastAPI already exposes `/metrics` and runs on host port 8000.
- Local development only: start FastAPI with `--host 0.0.0.0` to allow Docker Desktop access. Restrict firewall access; do not expose the API publicly.

## Start
From repository project root:

```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

In a second PowerShell terminal:

```powershell
docker compose --env-file .env.monitoring -f docker-compose.monitoring.yml config
docker compose --env-file .env.monitoring -f docker-compose.monitoring.yml up -d
docker compose --env-file .env.monitoring -f docker-compose.monitoring.yml ps
```

## Verify
- FastAPI: `http://127.0.0.1:8000/metrics`
- Prometheus: `http://127.0.0.1:9090/targets` (neojaundice-api should be UP)
- Grafana: `http://127.0.0.1:3000` (credentials in `.env.monitoring`)
- Dashboard: Dashboards > NeoJaundice > NeoJaundice API Overview.

Generate traffic by calling `/api/v1/health` repeatedly. Inference graphs require prediction requests; a zero or blank graph does not necessarily indicate a fault. Some graphs need multiple scrape intervals before rates appear.

## Troubleshooting
- Prometheus target DOWN: check FastAPI is running, host port 8000, Docker Desktop and Windows firewall. From the host, test `Invoke-WebRequest http://127.0.0.1:8000/metrics`.
- Dashboard empty: check Prometheus target, datasource and wait 30-60 seconds while generating traffic.
- Grafana credentials: read `.env.monitoring`. The password is generated once and not overwritten by re-running setup.
- Port already allocated: stop the other local service or change the localhost port mapping.
- Docker unavailable: start Docker Desktop and check `docker info`.

## Stop
```powershell
docker compose --env-file .env.monitoring -f docker-compose.monitoring.yml down
```

Do not use `down -v` unless you intend to delete saved metrics and dashboard state.

## Privacy and safety
Keep services on localhost. Never put patient IDs, images, names, or clinical values in Prometheus labels. Keep `.env.monitoring` out of Git. Do not publicly expose `/metrics` in production without access controls.

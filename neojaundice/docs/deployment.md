# Deployment runbook

## Local Docker Compose

The frontend is built in a Node stage and served by Nginx. Nginx proxies `/api/` to the Compose service `api:8000`; the browser uses the same origin and does not need a baked-in API URL.

```powershell
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml config
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml up --build -d
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml ps
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml logs -f frontend api
```

Frontend: `http://localhost:8080`. API host port: `http://127.0.0.1:8001`.

```powershell
docker compose --env-file .env.docker -f docker-compose.api.yml -f docker-compose.frontend.yml down
```

## API upstream override

`NEOJAUNDICE_API_UPSTREAM` can override the default `http://api:8000` when the backend service/network name differs. The frontend and backend must share a Docker network. Do not set the browser-side `VITE_API_BASE_URL` to a private container hostname; Nginx performs the proxy.

## Production hardening checklist

- Serve behind HTTPS and a managed reverse proxy/load balancer.
- Restrict CORS and network exposure to intended origins.
- Use secrets management; never bake secrets into images.
- Pin and scan base images and dependencies.
- Add resource limits, structured logs, request IDs, rate limiting, and monitoring.
- Decide data-retention policy before storing uploads or results; default to not retaining identifiable newborn images.
- Validate model checkpoint provenance, startup readiness, and inference parity.
- Do not expose the research prototype as a clinical decision tool.

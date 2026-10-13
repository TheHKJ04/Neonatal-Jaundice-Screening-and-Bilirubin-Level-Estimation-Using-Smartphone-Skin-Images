# NeoJaundice Docker Development Guide

## Purpose

Docker provides isolated development and deployment environments.

The initial Docker Compose stack contains:

- Python 3.13 verification container
- PostgreSQL 17 database container

## Prerequisites

- Docker Desktop or compatible Docker Engine
- Docker Compose
- Local `.env` file with DB_PASSWORD configured

## Verify Docker

```powershell
docker --version
docker compose version
docker info
```

## Validate Compose

```powershell
docker compose config --quiet
```

## Build the Python Image

```powershell
docker compose build app-check
```

## Start PostgreSQL

```powershell
docker compose up -d postgres
```

## Check PostgreSQL Health

```powershell
docker compose ps
```

## Run the Container Smoke Test

```powershell
docker compose run --rm app-check
```

Expected:

```text
[PASSED] Container environment verified.
```

## Stop Services

```powershell
docker compose down
```

The named database volume remains available.

## Important

Do not use:

```powershell
docker compose down -v
```

unless you intentionally want to delete the PostgreSQL volume.

## Security

Never commit `.env` or real credentials.

Do not include patient-identifiable data in Docker images.

This Docker configuration is intended for local development,
not clinical production deployment.

# NeoJaundice Deployment Guide

## Status

**Production deployment is not implemented.**

## 1. Current Infrastructure

The development environment currently includes:

- Python 3.13
- Docker
- Docker Compose
- PostgreSQL development container
- Container smoke test

## 2. Planned Services

Future deployment may include:

- React frontend
- FastAPI backend
- PostgreSQL
- Model inference service
- Monitoring components

## 3. Deployment Environments

### Development

Local experimentation and testing.

### Staging

Integration testing in a controlled environment.

### Production

Requires additional security, reliability,
monitoring, and clinical governance work.

## 4. Planned Container Strategy

Separate containers for:

- Backend
- Frontend
- Database
- Monitoring

A dedicated inference service may be introduced
if scaling requirements justify it.

## 5. Configuration

Use environment-specific configuration.

Do not bake secrets into container images.

## 6. Database Migrations

Alembic is planned for schema migration management.

Migrations must be tested before deployment.

## 7. Health Checks

Services should expose:

- Liveness checks
- Readiness checks

## 8. Logging

Use structured application logs.

Avoid logging raw medical images or identifiable
patient information.

## 9. Monitoring

Planned tools:

- Prometheus
- Grafana

Potential metrics:

- Request latency
- Error rates
- Service availability
- Model inference time
- Image quality rejection rates

## 10. Security

Production deployment requires:

- HTTPS
- Authentication
- Authorization
- Secret management
- Restricted database access
- Backup and recovery procedures
- Appropriate data retention controls

## 11. Deployment Verification

Future deployment checks will include:

- Container health
- Database connectivity
- API contract tests
- Migration checks
- Monitoring checks
- Security review

## 12. Current Limitation

The existing Docker setup is for development
verification and is not a clinical deployment.

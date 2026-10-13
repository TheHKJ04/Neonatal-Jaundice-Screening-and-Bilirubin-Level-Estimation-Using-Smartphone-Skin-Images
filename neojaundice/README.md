# NeoJaundice

## Neonatal Jaundice Screening and Bilirubin Level Estimation Using Smartphone Skin Images

> **Research and development project — not a clinically validated medical device.**

NeoJaundice is a computer vision research project investigating
whether smartphone skin images can support neonatal jaundice
screening and bilirubin level estimation.

The project aims to combine image processing, machine learning,
backend development, frontend development, and MLOps into a
reproducible research platform.

## Project Objectives

1. Develop a validated dataset preparation pipeline.
2. Investigate image preprocessing and skin-region extraction.
3. Compare traditional machine learning and deep learning models.
4. Explore EfficientNet-based image representations.
5. Investigate bilirubin regression using paired reference measurements.
6. Evaluate model performance across relevant patient and acquisition groups.
7. Develop a FastAPI backend and React frontend.
8. Introduce reproducible ML experiments and model versioning.
9. Containerize and monitor the application.
10. Document research findings and limitations.

## Current Development Status

**Phase 1: Engineering Foundation**

Implemented and locally verified:

- Modular repository structure
- Python 3.13 development environment
- Dependency configuration
- Pydantic application settings
- Hydra configuration structure
- VS Code development configuration
- Docker development image
- PostgreSQL development container
- Unit testing
- Ruff linting and formatting
- Pre-commit configuration
- GitHub Actions CI workflow definition

GitHub Actions execution must be verified separately.

The dataset pipeline, trained models, prediction API,
frontend, and clinical evaluation are not yet implemented.

## Planned Technology Stack

### Frontend

- React
- TypeScript
- Tailwind CSS

### Backend

- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic

### Computer Vision and Machine Learning

- Python
- PyTorch
- OpenCV
- Albumentations
- EfficientNet
- NumPy
- Pandas

### MLOps

- Docker
- Docker Compose
- MLflow
- DVC
- GitHub Actions

### Monitoring

- Prometheus
- Grafana

## Proposed Architecture

```text
Smartphone Image
       |
       v
Image Quality Validation
       |
       v
Preprocessing
       |
       v
Skin Region Extraction
       |
       v
Feature Extraction
       |
       v
ML Model
       |
       +------------------+
       |                  |
       v                  v
Screening Output    Bilirubin Estimate
       |                  |
       +---------+--------+
                 |
                 v
          FastAPI Backend
                 |
                 v
          React Frontend
```

This is a proposed architecture, not an implemented
end-to-end prediction system.

## Quick Start

### Prerequisites

- Python 3.13
- Git
- Docker Engine with Docker Compose
- VS Code (recommended)

### Clone the Repository

```powershell
git clone <repository-url>
cd neojaundice
```

### Install Development Dependencies

```powershell
python -m pip install -e ".[dev]"
```

### Configure Environment Variables

```powershell
Copy-Item .env.example .env
```

Set a unique local database password in `.env`.

Do not commit `.env`.

### Run Unit Tests

```powershell
python -m pytest tests/unit -v
```

### Run Ruff

```powershell
python -m ruff check scripts backend ml tests
python -m ruff format --check scripts backend ml tests
```

### Verify Configuration

```powershell
python scripts/verify_config.py
```

### Run Docker Verification

```powershell
docker compose config --quiet
docker compose run --rm app-check
```

### Stop Containers

```powershell
docker compose down
```

## Documentation

Start with [Documentation Index](docs/index.md).

## Research Considerations

Model development requires particular attention to:

- Patient-level data separation
- Reference bilirubin measurements
- Skin pigmentation diversity
- Smartphone camera differences
- Lighting and exposure variation
- Image quality
- Age at measurement
- Clinical confounders
- Model calibration
- External validation

## Medical Disclaimer

NeoJaundice is intended for research and educational development.

Smartphone image-based bilirubin estimation must not be used
to diagnose, exclude, or manage neonatal hyperbilirubinemia
without appropriate clinical validation and regulatory review.

Clinical assessment and established bilirubin measurement
methods remain essential.

## License

A license has not yet been selected.

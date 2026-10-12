# NeoJaundice Configuration Management

## Overview

NeoJaundice uses two configuration systems:

1. Pydantic Settings for application configuration.
2. Hydra for machine learning experiment configuration.

## Application Configuration

Application settings are defined in:

`backend/app/core/config.py`

A template is provided in:

`.env.example`

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`.

## Hydra Configuration

The main configuration file is:

`configs/config.yaml`

Configuration groups:

- `configs/data/`
- `configs/model/`
- `configs/training/`
- `configs/inference/`
- `configs/deployment/`

## Verify Configuration

```powershell
python scripts/verify_config.py
```

## Run Tests

```powershell
python -m pytest tests/unit/test_configuration.py -v
```

## Reproducibility

Training runs should record:

- Git commit
- Dataset version
- Model configuration
- Training configuration
- Random seed
- Package versions
- Hardware information
- Evaluation metrics

Future MLflow integration will record experiment metadata.

## Security

Do not store real credentials in Hydra YAML files.

Use environment variables or an appropriate secrets manager.

Do not commit patient-identifiable medical data.

## Medical Safety

Configuration values are experimental defaults.

Clinical thresholds and treatment decisions must not be
derived from these defaults without appropriate clinical
validation and oversight.
# NeoJaundice Testing and Code Quality

## Overview

NeoJaundice uses:

- pytest for automated tests
- pytest-cov for coverage
- Ruff for linting and formatting
- mypy for static type checking
- pre-commit for local Git hooks
- GitHub Actions for continuous integration

## Run Unit Tests

```powershell
python -m pytest tests/unit -v
```

## Run All Tests

```powershell
python -m pytest -v
```

## Run Tests With Coverage

```powershell
python -m pytest tests/unit -v --cov=backend --cov=ml --cov=scripts --cov-report=term-missing
```

## Ruff Linting

```powershell
python -m ruff check scripts backend ml tests
```

## Ruff Formatting

```powershell
python -m ruff format --check scripts backend ml tests
```

To apply formatting:

```powershell
python -m ruff format scripts backend ml tests
```

## Pre-commit

Install:

```powershell
python -m pre_commit install
```

Run manually:

```powershell
python -m pre_commit run --all-files
```

## Configuration Verification

```powershell
python scripts/verify_quality_config.py
```

## CI

GitHub Actions runs quality checks for pushes and pull requests
targeting the configured branches.

## Testing Principles

1. Use deterministic unit tests.
2. Avoid patient-identifiable data in test fixtures.
3. Test edge cases and failure paths.
4. Separate unit tests from integration tests.
5. Use patient-level splits in ML evaluation.
6. Record dataset and model versions.
7. Do not interpret software test coverage as clinical validation.

## Medical Safety

NeoJaundice is a research system.

Passing software tests does not establish clinical accuracy,
safety, or suitability for medical decision-making.

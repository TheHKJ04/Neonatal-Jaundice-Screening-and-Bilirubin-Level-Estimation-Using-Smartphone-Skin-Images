# NeoJaundice — Troubleshooting Guide

## 1. Overview

This document provides solutions to common problems encountered while developing NeoJaundice.

**Current development environment:**

- Operating system: Windows
- Python: 3.13
- Terminal: PowerShell
- Editor: VS Code
- Testing: pytest
- Code quality: Ruff
- Configuration: Pydantic Settings and Hydra
- Containerization: Docker and Docker Compose
- Database: PostgreSQL

The current local development setup uses Python 3.13 without a virtual environment.

All commands should be executed from the NeoJaundice project root unless otherwise specified.

## 2. Python Troubleshooting

### 2.1 Python Version Is Incorrect

**Problem:** The terminal uses a different Python version.

**Verify:**

```powershell
python --version
```

Expected:

```text
Python 3.13.x
```

Check the Python executable:

```powershell
python -c "import sys; print(sys.executable)"
```

**Solution:**

Ensure Python 3.13 is installed and selected in VS Code.

Open the Command Palette using `Ctrl + Shift + P`, select `Python: Select Interpreter`, and choose Python 3.13.

### 2.2 ModuleNotFoundError

**Problem:**

```text
ModuleNotFoundError: No module named 'neojaundice'
```

**Cause:**

The repository directory name does not automatically become an importable Python package.

The current project uses top-level packages such as `backend`, `ml`, and `scripts`.

**Solution:**

Use imports that match the actual directory structure.

Correct:

```python
from scripts.container_smoke_test import REQUIRED_PACKAGES
```

Incorrect for the current architecture:

```python
from neojaundice.scripts.container_smoke_test import REQUIRED_PACKAGES
```

Verify the import:

```powershell
python -c "from scripts.container_smoke_test import REQUIRED_PACKAGES; print(REQUIRED_PACKAGES)"
```

### 2.3 Missing Python Dependencies

**Problem:**

```text
ModuleNotFoundError: No module named 'yaml'
```

**Solution:**

Install the declared development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Verify package consistency:

```powershell
python -m pip check
```

Check a specific package:

```powershell
python -m pip show PyYAML
```

Dependencies required by project tests should be declared in `pyproject.toml`, not installed manually without recording them.

### 2.4 Dependency Conflicts

**Problem:** Installing a package changes another package's version or causes incompatibilities.

**Cause:** The current local environment uses a shared Python installation.

**Solution:**

Inspect installed packages:

```powershell
python -m pip list
```

Check dependency consistency:

```powershell
python -m pip check
```

Avoid unnecessary global upgrades.

Use the Docker environment when a reproducible, isolated runtime is required.

## 3. Pytest Troubleshooting

### 3.1 No Tests Collected

**Problem:**

```text
collected 0 items
```

**Possible causes:**

- Incorrect test filename
- Incorrect test directory
- Missing test functions
- Incorrect pytest configuration

**Solution:**

Use filenames beginning with `test_`.

Example:

```text
tests/unit/test_configuration.py
```

Run:

```powershell
python -m pytest tests/unit -v
```

Inspect test discovery:

```powershell
python -m pytest tests/unit --collect-only -q
```

### 3.2 Tests Fail During Collection

**Problem:**

```text
ERROR collecting tests/unit/test_container_configuration.py
```

**Cause:** Pytest cannot import a module required by the test.

**Solution:**

Check the import path and confirm the referenced file exists.

For example:

```text
scripts/container_smoke_test.py
```

Should be imported using:

```python
from scripts.container_smoke_test import check_python_version
```

Run the failing module separately:

```powershell
python -m pytest tests/unit/test_container_configuration.py -v
```

### 3.3 Documentation Tests Fail

**Problem:**

```text
AssertionError: Documentation file is empty
```

**Cause:** A required documentation file exists but contains no content.

**Solution:**

Open the reported Markdown file and add meaningful documentation.

Do not remove the assertion merely to make the test pass.

Run:

```powershell
python -m pytest tests/unit/test_documentation.py -v
```

### 3.4 Test Count Is Different Than Expected

**Problem:** The number of collected tests differs from the previous run.

**Solution:**

Inspect test discovery:

```powershell
python -m pytest tests/unit --collect-only -q
```

Check for renamed, moved, or deleted test files.

A different test count is not automatically an error, but unexpected changes should be investigated.

## 4. Ruff Troubleshooting

### 4.1 Import Block Is Unsorted

**Problem:**

```text
I001 Import block is un-sorted or un-formatted
```

**Solution:**

```powershell
python -m ruff check scripts backend ml tests --fix
```

Verify:

```powershell
python -m ruff check scripts backend ml tests
```

### 4.2 Unused Imports

**Problem:**

```text
F401 imported but unused
```

**Solution:**

Remove the unused import or allow Ruff to apply the suggested fix.

```powershell
python -m ruff check scripts backend ml tests --fix
```

Review the changes before committing.

### 4.3 Formatting Check Fails

**Problem:**

```text
File would be reformatted
```

**Solution:**

Apply formatting:

```powershell
python -m ruff format scripts backend ml tests
```

Verify:

```powershell
python -m ruff format --check scripts backend ml tests
```

Run unit tests after automatic code modifications.

## 5. Git Troubleshooting

### 5.1 Git Tracks Ignored Files

**Problem:** Python cache files appear in Git even though `.gitignore` excludes them.

**Cause:** The files were already tracked before the ignore rules were added.

**Solution:**

Find tracked files matching ignore rules:

```powershell
git ls-files -ci --exclude-standard
```

Remove a specific file from Git tracking:

```powershell
git rm --cached -- "path/to/file"
```

This preserves the local file.

Review the staged changes before committing.

### 5.2 Python Cache Files Are Tracked

**Problem:**

```text
scripts/__pycache__/verify_structure.cpython-313.pyc
```

**Solution:**

Ensure `.gitignore` contains:

```gitignore
__pycache__/
*.py[cod]
```

Remove previously tracked cache files using `git rm --cached`.

### 5.3 Environment File Appears in Git

**Problem:** `.env` appears in Git changes.

**Solution:**

Ensure `.gitignore` contains:

```gitignore
.env
.env.*
!.env.example
```

Verify:

```powershell
git check-ignore .env
```

If `.env` is already tracked:

```powershell
git rm --cached -- .env
```

If credentials were previously committed, assess exposure and rotate affected credentials.

Removing the file from the latest commit does not erase older Git history.

### 5.4 Git Commit Is Blocked

**Problem:** Pre-commit checks fail during `git commit`.

**Solution:**

Run:

```powershell
python -m pre_commit run --all-files
```

Review any automatic changes:

```powershell
git diff
```

Stage corrected files and retry the commit.

Do not bypass quality checks without understanding the failure.

## 6. Docker Troubleshooting

### 6.1 Docker Daemon Is Not Running

**Problem:**

```text
Cannot connect to the Docker daemon
```

**Solution:**

Start Docker Desktop.

Verify:

```powershell
docker info
```

### 6.2 Docker Compose Configuration Is Invalid

**Problem:** Docker Compose reports a configuration error.

**Solution:**

Run:

```powershell
docker compose config --quiet
```

Check YAML indentation, environment variables, service definitions, and volume configuration.

### 6.3 Database Password Is Missing

**Problem:**

```text
Set DB_PASSWORD in your local .env file
```

**Solution:**

Open the local `.env` file and configure a nonempty development database password.

Do not commit or share the password.

### 6.4 PostgreSQL Is Not Healthy

**Problem:** PostgreSQL remains in a starting or unhealthy state.

**Solution:**

Inspect services:

```powershell
docker compose ps
```

Inspect logs:

```powershell
docker compose logs postgres
```

Check that the configured database environment variables are valid.

### 6.5 PostgreSQL Port Is Already Used

**Problem:**

```text
port is already allocated
```

**Solution:**

Check port 5432:

```powershell
Get-NetTCPConnection -LocalPort 5432 -ErrorAction SilentlyContinue
```

If another application uses the port, change the host port mapping in `compose.yaml`.

Example:

```yaml
ports:
  - "127.0.0.1:5433:5432"
```

### 6.6 Container Smoke Test Fails

**Problem:** The Python container reports a missing dependency or incorrect runtime.

**Solution:**

Rebuild:

```powershell
docker compose build app-check
```

Run:

```powershell
docker compose run --rm app-check
```

Inspect the build output for dependency installation errors.

### 6.7 Stop Docker Containers Safely

Use:

```powershell
docker compose down
```

This normally preserves named volumes.

Avoid:

```powershell
docker compose down -v
```

unless deleting the database volume is intentional.

## 7. Application Configuration Troubleshooting

### 7.1 Pydantic Validation Error

**Problem:** Application settings fail validation.

**Possible causes:**

- Invalid port number
- Unsupported environment value
- Incorrect environment variable format
- Invalid JSON for list-valued settings

**Solution:**

Review `.env` and `backend/app/core/config.py`.

Run:

```powershell
python scripts/verify_config.py
```

### 7.2 Hydra Configuration Error

**Problem:** Hydra cannot compose the configuration.

**Possible causes:**

- Missing YAML file
- Incorrect configuration group
- Invalid defaults list
- YAML syntax error

**Solution:**

Verify the `configs/` directory and its configuration groups.

Run:

```powershell
python scripts/verify_config.py
```

## 8. Pre-commit Troubleshooting

### 8.1 Pre-commit Is Not Installed

**Problem:**

```text
No module named pre_commit
```

**Solution:**

Install the project's development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Then install the Git hook:

```powershell
python -m pre_commit install
```

### 8.2 Pre-commit Modifies Files

**Problem:** A formatting hook reports that files were modified.

**Solution:**

Review changes:

```powershell
git diff
```

Rerun:

```powershell
python -m pre_commit run --all-files
```

Stage the corrected files before committing.

### 8.3 Ruff Versions Produce Different Formatting

**Problem:** Local Ruff and pre-commit Ruff disagree.

**Cause:** Different tool versions may have different formatting behavior.

**Solution:**

Compare the configured pre-commit Ruff version with the locally installed version.

Align tool versions through the project's dependency configuration.

## 9. GitHub Actions Troubleshooting

### 9.1 Workflow Does Not Run

**Possible causes:**

- Workflow file is not committed
- Branch does not match the configured trigger
- Workflow YAML is invalid
- Repository Actions settings restrict execution

**Solution:**

Verify:

```text
.github/workflows/ci.yml
```

Commit and push the workflow.

Check the repository's Actions tab.

### 9.2 Tests Pass Locally but Fail in CI

**Possible causes:**

- Different dependency versions
- Missing dependencies
- Operating system differences
- Case-sensitive filesystem paths
- Missing configuration
- Environment-specific assumptions

**Solution:**

Inspect the failing GitHub Actions step.

Reproduce the failure using the same Python version and declared dependencies.

Do not modify tests merely to hide CI failures.

## 10. Coverage Troubleshooting

### 10.1 Coverage Report Is Not Generated

Verify pytest-cov is installed:

```powershell
python -m pip show pytest-cov
```

Run:

```powershell
python -m pytest tests/unit --cov=backend --cov=ml --cov=scripts --cov-report=term-missing --cov-report=xml
```

### 10.2 Coverage Is Lower Than Expected

Low coverage indicates that some code paths were not executed during testing.

Identify untested modules and add meaningful tests.

Do not write tests solely to increase the percentage.

## 11. Security and Healthcare Data

### 11.1 Patient Data Accidentally Added to Git

If identifiable patient data is accidentally committed:

1. Stop further distribution.
2. Assess whether the data was exposed.
3. Follow applicable privacy and incident-response procedures.
4. Review Git history.
5. Remove inappropriate repository content using an approved remediation process.
6. Document the incident and corrective actions.

Do not publish medical images or clinical records without appropriate authorization.

### 11.2 Sensitive Files in Docker Images

Ensure `.dockerignore` excludes private datasets, credentials, and unnecessary local artifacts.

Do not copy patient-identifiable datasets into development images.

## 12. General Debugging Procedure

When an error occurs:

1. Read the complete error message.
2. Identify the first meaningful exception.
3. Determine which component failed.
4. Reproduce the issue using the smallest relevant command.
5. Inspect the affected file or configuration.
6. Apply a focused fix.
7. Rerun the failing check.
8. Run the full regression suite.
9. Update documentation if necessary.
10. Commit the verified fix.

## 13. Verification Commands

Run these commands from the repository root.

### Python

```powershell
python --version
```

### Dependencies

```powershell
python scripts/verify_dependencies.py
python -m pip check
```

### Configuration

```powershell
python scripts/verify_config.py
python scripts/verify_quality_config.py
```

### Code Quality

```powershell
python -m ruff check scripts backend ml tests
python -m ruff format --check scripts backend ml tests
```

### Tests

```powershell
python -m pytest tests/unit -v
```

### Coverage

```powershell
python -m pytest tests/unit --cov=backend --cov=ml --cov=scripts --cov-report=term-missing
```

### Docker

```powershell
docker compose config --quiet
docker compose run --rm app-check
```

### Git

```powershell
git status --short
git ls-files -ci --exclude-standard
```

## 14. Maintenance

Update this guide whenever a new recurring development problem is identified.

Each new troubleshooting entry should contain:

- Error message
- Root cause
- Resolution
- Verification command
- Preventive action, where applicable

**Last reviewed:** October 2026

**Project status:** Engineering foundation under development.

**Clinical status:** Research only; not clinically validated.

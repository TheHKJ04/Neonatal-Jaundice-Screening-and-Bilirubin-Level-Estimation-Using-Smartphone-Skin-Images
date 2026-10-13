# NeoJaundice Development Setup

## Requirements

- Windows 10/11, Linux, or macOS
- Python 3.13
- Git
- VS Code
- Docker Desktop (for later phases)
- Node.js (for frontend development)

## 1. Clone the Repository

Once the remote repository is available:

```bash
git clone <repository-url>
cd neojaundice
```

## 2. Create a Virtual Environment

Windows:

```powershell
py -3.13 -m venv .venv
```

Linux/macOS:

```bash
python3.13 -m venv .venv
```

## 3. Activate the Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 5. Install Development Dependencies

```bash
python -m pip install -e ".[dev]"
```

## 6. Verify Dependencies

```bash
python scripts/verify_dependencies.py
```

## 7. Run Tests

```bash
python -m pytest tests/unit -v
```

## 8. Verify Project Structure

```bash
python scripts/verify_structure.py
```

## Optional Dependency Groups

Backend:

```bash
python -m pip install -e ".[backend]"
```

Computer vision:

```bash
python -m pip install -e ".[cv]"
```

Machine learning:

```bash
python -m pip install -e ".[ml]"
```

MLOps:

```bash
python -m pip install -e ".[mlops]"
```

Monitoring:

```bash
python -m pip install -e ".[monitoring]"
```

Note: GPU-enabled PyTorch installation requires platform-specific
configuration and should be performed after hardware verification.

## Security

Never commit:

- `.env`
- Patient images
- Clinical records
- Credentials
- Private keys
- Model checkpoints containing sensitive information

## Medical Disclaimer

NeoJaundice is a research project.

It is not a clinically validated medical device and must not be used
to make independent medical decisions.
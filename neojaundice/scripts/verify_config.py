"""
Verify NeoJaundice application and Hydra configurations.

Run:
    python scripts/verify_config.py
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from backend.app.core.config import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"


def verify_application_settings() -> bool:
    """Validate application settings."""

    settings = Settings()

    print("[OK] Application settings loaded")
    print(f"     Application: {settings.app_name}")
    print(f"     Environment: {settings.app_env}")
    print(f"     API port: {settings.api_port}")

    return True


def verify_hydra_configuration() -> bool:
    """Validate Hydra configuration composition."""

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base=None,
    ):
        config = compose(config_name="config")

    required_sections = (
        "data",
        "model",
        "training",
        "inference",
        "deployment",
    )

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing Hydra configuration section: {section}"
            )

    # Convert the composed configuration to a plain
    # Python container to verify that its structure
    # can be accessed successfully.
    OmegaConf.to_container(
        config,
        resolve=False,
    )

    print("[OK] Hydra configuration composed")
    print(f"     Model: {config.model.architecture}")
    print(f"     Epochs: {config.training.epochs}")
    print(f"     Batch size: {config.training.batch_size}")

    return True


def main() -> int:
    """Run configuration verification."""

    print("=" * 55)
    print("NeoJaundice Configuration Verification")
    print("=" * 55)

    try:
        verify_application_settings()
        verify_hydra_configuration()

    except Exception as exc:
        print(f"\n[FAILED] {type(exc).__name__}: {exc}")
        return 1

    print("\n[PASSED] Configuration verification successful.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
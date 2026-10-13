"""
Verify NeoJaundice dataset source registry.

Usage:
    python scripts/verify_dataset_sources.py
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ml.datasets.source_registry import (
    load_dataset_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = PROJECT_ROOT / "configs" / "data" / "sources.yaml"


def main() -> int:
    """Validate and summarize registered dataset sources."""

    print("=" * 60)
    print("NeoJaundice Dataset Source Verification")
    print("=" * 60)

    try:
        registry = load_dataset_registry(REGISTRY_PATH)

    except (
        FileNotFoundError,
        OSError,
        ValueError,
        ValidationError,
    ) as exc:
        print(f"[FAILED] Dataset registry: {exc}")
        return 1

    print(f"[PASSED] Registry version: {registry.registry_version}")

    print(f"[PASSED] Registered datasets: {len(registry.datasets)}")

    for dataset in registry.datasets:
        print()
        print(f"Dataset: {dataset.dataset_id}")
        print(f"  Provider: {dataset.provider}")
        print(f"  Access: {dataset.access_status.value}")
        print(f"  License: {dataset.license_status}")

        tasks = ", ".join(task.value for task in dataset.task_types)

        print(f"  Tasks: {tasks}")

    print()
    print("[PASSED] Dataset source registry validated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

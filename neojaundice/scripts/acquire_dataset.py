"""
Command-line interface for controlled dataset acquisition.

Example:
    python -m scripts.acquire_dataset \
        --dataset-id example_dataset \
        --archive data/external/example_dataset/source_archive.zip \
        --destination data/raw/example_dataset \
        --sha256 EXPECTED_HASH
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ml.datasets.acquisition import DatasetApproval
from ml.datasets.acquisition_workflow import acquire_local_archive
from ml.datasets.source_registry import load_dataset_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_REGISTRY = PROJECT_ROOT / "configs/data/sources.yaml"

APPROVAL_REGISTRY = PROJECT_ROOT / "configs/data/acquisition_approvals.yaml"

AUDIT_LOG = PROJECT_ROOT / "logs/dataset_acquisition.jsonl"


class ApprovalRegistry(BaseModel):
    """Collection of acquisition approvals."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)

    approvals: list[DatasetApproval]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire an approved local dataset archive."
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
    )

    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sha256",
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    sources = load_dataset_registry(SOURCE_REGISTRY)

    source_ids = {dataset.dataset_id for dataset in sources.datasets}

    if args.dataset_id not in source_ids:
        print("[FAILED] Dataset is not registered.")
        return 1

    if not APPROVAL_REGISTRY.is_file():
        print("[FAILED] Private approval registry not found.")
        return 1

    with APPROVAL_REGISTRY.open(
        "r",
        encoding="utf-8",
    ) as file:
        approval_data = yaml.safe_load(file)

    approval_registry = ApprovalRegistry.model_validate(approval_data)

    approvals = {
        approval.dataset_id: approval for approval in approval_registry.approvals
    }

    approval = approvals.get(args.dataset_id)

    if approval is None:
        print("[FAILED] No approval record for dataset.")
        return 1

    try:
        result = acquire_local_archive(
            approval=approval,
            archive_path=args.archive,
            destination=args.destination,
            expected_sha256=args.sha256,
            audit_log=AUDIT_LOG,
        )

    except (
        PermissionError,
        FileNotFoundError,
        ValueError,
        OSError,
    ) as exc:
        print(f"[FAILED] {exc}")
        return 1

    print("=" * 60)
    print("NeoJaundice Dataset Acquisition")
    print("=" * 60)

    print(f"Dataset: {result.dataset_id}")
    print(f"Destination: {result.destination}")
    print(f"Extracted files: {result.extracted_files}")
    print(f"SHA-256 verified: {result.verified_sha256}")

    print()
    print("[PASSED] Dataset acquisition completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

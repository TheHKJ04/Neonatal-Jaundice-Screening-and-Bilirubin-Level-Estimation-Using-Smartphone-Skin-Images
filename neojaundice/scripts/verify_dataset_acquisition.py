"""
Verify dataset acquisition configuration.

Usage:
    python scripts/verify_dataset_acquisition.py
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ml.datasets.acquisition import DatasetApproval

PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPROVAL_TEMPLATE = (
    PROJECT_ROOT / "configs" / "data" / "acquisition_approvals.example.yaml"
)


class ApprovalRegistry(BaseModel):
    """Collection of dataset acquisition approvals."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)

    approvals: list[DatasetApproval]


def main() -> int:
    """Validate the acquisition approval template."""

    with APPROVAL_TEMPLATE.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_data = yaml.safe_load(file)

    registry = ApprovalRegistry.model_validate(raw_data)

    print("=" * 60)
    print("NeoJaundice Dataset Acquisition Verification")
    print("=" * 60)

    print(f"Schema version: {registry.schema_version}")

    print(f"Approval records: {len(registry.approvals)}")

    for approval in registry.approvals:
        print(f"{approval.dataset_id}: {approval.status.value}")

    print()
    print("[PASSED] Acquisition template is valid.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

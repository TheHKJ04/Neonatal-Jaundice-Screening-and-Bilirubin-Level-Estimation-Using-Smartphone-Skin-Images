# type: ignore
"""
Integrate image inspection and clinical metadata.

Usage:
    python -m scripts.integrate_metadata
"""

from pathlib import Path

from ml.datasets.metadata_mapper import (
    integrate_metadata,
    write_standardized_manifest,
)

CLINICAL_CSV = Path("data/raw/synthetic_demo/clinical_metadata.csv")

IMAGE_MANIFEST = Path("data/interim/synthetic_demo_inspection/image_manifest.csv")

OUTPUT_CSV = Path("data/processed/synthetic_demo/standardized_manifest.csv")


def main() -> int:
    """Execute synthetic metadata integration."""

    result = integrate_metadata(
        clinical_csv=CLINICAL_CSV,
        image_manifest_csv=IMAGE_MANIFEST,
    )

    print("=" * 60)
    print("NeoJaundice Clinical Metadata Integration")
    print("=" * 60)

    print(f"Total rows: {result.total_rows}")
    print(f"Accepted: {result.accepted_rows}")
    print(f"Rejected: {result.rejected_rows}")

    for error in result.errors:
        print(f"[ERROR] {error}")

    if result.rejected_rows:
        print("[FAILED] Some metadata records were rejected.")
        return 1

    write_standardized_manifest(
        result.records,
        OUTPUT_CSV,
    )

    print()
    print(f"Output: {OUTPUT_CSV}")
    print("[PASSED] Metadata integration completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

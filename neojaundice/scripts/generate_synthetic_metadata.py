# type: ignore
"""
Generate synthetic clinical metadata for pipeline testing.

Values are artificial and must not be used for
clinical research conclusions.
"""

import csv
from pathlib import Path

OUTPUT_PATH = Path("data/raw/synthetic_demo/clinical_metadata.csv")


def main() -> None:
    """Generate synthetic metadata records."""

    records = []

    for index in range(1, 5):
        records.append(
            {
                "sample_id": f"S{index:03d}",
                "patient_id": f"P{index:03d}",
                "image_path": (f"synthetic_{index:03d}.png"),
                "captured_at": ("2026-01-10T10:00:00+00:00"),
                "bilirubin_measured_at": ("2026-01-10T10:30:00+00:00"),
                "bilirubin_type": "TSB",
                "bilirubin_mg_dl": (7.0 + index),
                "age_hours": 48.0 + index,
                "gestational_age_weeks": 38.0,
                "body_region": "forehead",
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )

        writer.writeheader()

        writer.writerows(records)

    print(f"Synthetic clinical metadata created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

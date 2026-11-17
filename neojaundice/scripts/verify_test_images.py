from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data/processed/neojaundice/test.csv"

if not MANIFEST.exists():
    raise FileNotFoundError(f"Test manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)

print("Test records:", len(df))
print("Unique patients:", df["patient_id"].nunique())
print("Image path examples:")
print(df["image_path"].head(5).to_list())

# Possible dataset roots. We will verify them rather than assume one.
candidate_roots = [
    PROJECT_ROOT,
    PROJECT_ROOT / "data/external/neojaundice/extracted/NeoJaundice",
    PROJECT_ROOT / "data/external/neojaundice",
]

print("\nCandidate image root checks:")

for root in candidate_roots:
    existing = sum(
        (root / str(relative_path)).is_file() for relative_path in df["image_path"]
    )

    print(f"{root}\n  Existing images: {existing}/{len(df)}")

# Search by image filename as a fallback.
print("\nSearching for sample image filenames:")

for filename in df["image_id"].head(5):
    matches = list(PROJECT_ROOT.rglob(str(filename)))

    print(f"\n{filename}:")
    for match in matches[:5]:
        print(" ", match)

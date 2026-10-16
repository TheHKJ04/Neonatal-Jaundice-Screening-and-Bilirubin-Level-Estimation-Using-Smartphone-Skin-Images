"""NeoJaundice dataset adaptation and patient-level splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {
    "patient_id",
    "image_idx",
    "gender",
    "gestational_age",
    "age(day)",
    "weight",
    "blood(mg/dL)",
    "Treatment",
}


def prepare_manifest(
    dataset_root: Path,
) -> pd.DataFrame:
    """Load and validate NeoJaundice metadata."""

    dataset_root = Path(dataset_root).resolve()

    csv_path = dataset_root / "chd_jaundice_published_2.csv"

    image_directory = dataset_root / "images"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_directory}")

    dataframe = pd.read_csv(
        csv_path,
        dtype={
            "patient_id": str,
            "image_idx": str,
        },
    )

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if dataframe.empty:
        raise ValueError("Dataset is empty.")

    if dataframe["patient_id"].isna().any():
        raise ValueError("Missing patient identifiers.")

    if dataframe["image_idx"].isna().any():
        raise ValueError("Missing image identifiers.")

    if dataframe["image_idx"].duplicated().any():
        raise ValueError("Duplicate image identifiers.")

    bilirubin = pd.to_numeric(
        dataframe["blood(mg/dL)"],
        errors="coerce",
    )

    if bilirubin.isna().any():
        raise ValueError("Missing or invalid bilirubin values.")

    if not np.isfinite(bilirubin.to_numpy()).all():
        raise ValueError("Non-finite bilirubin values.")

    if (bilirubin < 0).any():
        raise ValueError("Negative bilirubin values.")

    treatment = pd.to_numeric(
        dataframe["Treatment"],
        errors="coerce",
    )

    if treatment.isna().any():
        raise ValueError("Invalid treatment labels.")

    if not treatment.isin([0, 1]).all():
        raise ValueError("Treatment labels must be binary.")

    patient_ids = dataframe["patient_id"].astype(str)

    image_names = dataframe["image_idx"].astype(str)

    image_paths = []

    for image_name in image_names:
        candidate = Path(image_name)

        if (
            candidate.is_absolute()
            or candidate.name != image_name
            or image_name in {".", ".."}
        ):
            raise ValueError(f"Unsafe image filename: {image_name}")

        image_path = (image_directory / image_name).resolve()

        if image_directory.resolve() not in image_path.parents:
            raise ValueError(f"Image escapes dataset directory: {image_name}")

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_name}")

        image_paths.append(image_path.relative_to(dataset_root).as_posix())

    age_days = pd.to_numeric(
        dataframe["age(day)"],
        errors="coerce",
    )

    gestational_age = pd.to_numeric(
        dataframe["gestational_age"],
        errors="coerce",
    )

    weight = pd.to_numeric(
        dataframe["weight"],
        errors="coerce",
    )

    manifest = pd.DataFrame(
        {
            "image_id": image_names,
            "patient_id": patient_ids,
            "image_path": image_paths,
            "bilirubin_value": bilirubin.astype(float),
            "bilirubin_unit": "mg/dL",
            "measurement_type": "unspecified_blood",
            "treatment_label": treatment.astype(int),
            "gender": dataframe["gender"],
            "gestational_age_weeks": gestational_age,
            "postnatal_age_days": age_days,
            "weight_grams": weight,
            "zero_bilirubin_review": bilirubin.eq(0),
            "missing_age_review": age_days.isna(),
        }
    )

    # One patient must not have contradictory treatment labels.
    patient_label_counts = manifest.groupby("patient_id")["treatment_label"].nunique()

    if (patient_label_counts > 1).any():
        raise ValueError("Conflicting treatment labels within a patient.")

    return manifest.sort_values("image_id").reset_index(drop=True)


def create_patient_splits(
    manifest: pd.DataFrame,
    random_seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Create patient-disjoint stratified dataset splits."""

    fractions = (
        train_fraction,
        validation_fraction,
        test_fraction,
    )

    if any(value <= 0 or value >= 1 for value in fractions):
        raise ValueError("Split fractions must be between zero and one.")

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1.")

    required = {"patient_id", "treatment_label"}

    if not required.issubset(manifest.columns):
        raise ValueError("Manifest lacks patient or treatment columns.")

    if manifest["patient_id"].isna().any():
        raise ValueError("Missing patient identifiers.")

    if not manifest["treatment_label"].isin([0, 1]).all():
        raise ValueError("Invalid treatment labels.")

    patient_labels = (
        manifest.groupby("patient_id")["treatment_label"]
        .agg(["first", "nunique"])
        .reset_index()
    )

    if (patient_labels["nunique"] != 1).any():
        raise ValueError("Inconsistent treatment labels within a patient.")

    patients = patient_labels["patient_id"].to_numpy()
    labels = patient_labels["first"].to_numpy()

    train_patients, remaining_patients = train_test_split(
        patients,
        train_size=train_fraction,
        random_state=random_seed,
        stratify=labels,
    )

    remaining_labels = (
        patient_labels.set_index("patient_id")
        .loc[remaining_patients, "first"]
        .to_numpy()
    )

    relative_validation_fraction = validation_fraction / (
        validation_fraction + test_fraction
    )

    val_patients, test_patients = train_test_split(
        remaining_patients,
        train_size=relative_validation_fraction,
        random_state=random_seed,
        stratify=remaining_labels,
    )

    patient_groups = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }

    if (
        patient_groups["train"] & patient_groups["val"]
        or patient_groups["train"] & patient_groups["test"]
        or patient_groups["val"] & patient_groups["test"]
    ):
        raise AssertionError("Patient leakage detected.")

    splits = {}

    for split_name, patient_set in patient_groups.items():
        subset = manifest[manifest["patient_id"].isin(patient_set)].copy()

        splits[split_name] = subset.sort_values("image_id").reset_index(drop=True)

    return splits

"""Evaluate a selected NeoJaundice regression checkpoint on the held-out test split.

This script never trains or tunes a model. It expects the existing project modules:
- ml.training.dataset.NeoJaundiceDataset
- ml.training.model.build_model

Outputs are created automatically under artifacts/evaluation/regression by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a NeoJaundice regression checkpoint on the held-out test set."
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed/neojaundice"))
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/external/neojaundice/extracted/NeoJaundice"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best_model.pt for the selected regression run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/regression"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-column",
        default="blood(mg/dL)",
        help="Target column in test.csv. Used to document target semantics; dataset adapter supplies target values.",
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=1000,
        help="Patient-level bootstrap replicates; use 0 to skip confidence intervals.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--allow-checkpoint-metadata-mismatch",
        action="store_true",
        help="Allow checkpoint wrappers with metadata that differs from expected. Use only after inspecting the file.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    """Support common checkpoint layouts without silently accepting unknown layouts."""
    if isinstance(checkpoint_obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint_obj.get(key)
            if isinstance(candidate, dict) and candidate and all(
                isinstance(v, torch.Tensor) for v in candidate.values()
            ):
                return candidate
        if checkpoint_obj and all(
            isinstance(v, torch.Tensor) for v in checkpoint_obj.values()
        ):
            return checkpoint_obj
    raise ValueError(
        "Unsupported checkpoint structure. Expected a state_dict or a dictionary "
        "containing model_state_dict/state_dict. Inspect best_model.pt before proceeding."
    )


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for an empty prediction set.")
    return {
        "count": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_target": float(np.mean(y_true)),
        "mean_prediction": float(np.mean(y_pred)),
        "bias_prediction_minus_target": float(np.mean(y_pred - y_true)),
    }


def patient_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "target", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    # All images for one infant should have the same target. Average labels defensively
    # and average predictions to obtain one estimate per infant.
    return (
        frame.groupby("patient_id", dropna=False)
        .agg(target=("target", "mean"), prediction=("prediction", "mean"), image_count=("prediction", "size"))
        .reset_index()
    )


def bootstrap_patient_mae(
    patient_frame: pd.DataFrame, replicates: int, seed: int
) -> dict[str, float | int] | None:
    if replicates <= 0:
        return None
    if len(patient_frame) < 2:
        raise ValueError("At least two patients are required for bootstrap confidence intervals.")
    rng = np.random.default_rng(seed)
    targets = patient_frame["target"].to_numpy(dtype=float)
    predictions = patient_frame["prediction"].to_numpy(dtype=float)
    values = np.empty(replicates, dtype=float)
    n = len(patient_frame)
    for i in range(replicates):
        indices = rng.integers(0, n, size=n)
        values[i] = np.mean(np.abs(predictions[indices] - targets[indices]))
    return {
        "replicates": int(replicates),
        "confidence_level": 0.95,
        "patient_mae_ci_lower": float(np.quantile(values, 0.025)),
        "patient_mae_ci_upper": float(np.quantile(values, 0.975)),
    }


def load_test_dataset(manifest_dir: Path, image_root: Path):
    # Import from the existing project so preprocessing remains consistent with training.
    from ml.training.dataset import NeoJaundiceDataset

    test_manifest = manifest_dir / "test.csv"
    if not test_manifest.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")
    dataset = NeoJaundiceDataset(
        manifest=test_manifest,
        image_root=image_root,
        task="regression",
        preprocessing="resize",
        training=False,
        exclude_zero=False,
    )
    return dataset


def load_model(checkpoint_path: Path, device: torch.device):
    from ml.training.model import build_model

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Checkpoints created by this project are trusted local artifacts.
    try:
        checkpoint_obj = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch versions
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")

    state_dict = extract_state_dict(checkpoint_obj)
    model = build_model(pretrained=False)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch.\n"
            f"Missing keys: {incompatible.missing_keys[:20]}\n"
            f"Unexpected keys: {incompatible.unexpected_keys[:20]}\n"
            "Do not continue until the checkpoint architecture is verified."
        )
    model.to(device)
    model.eval()
    return model


def make_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    actual = predictions["target"].to_numpy(dtype=float)
    predicted = predictions["prediction"].to_numpy(dtype=float)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))

    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.65)
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual target")
    plt.ylabel("Predicted target")
    plt.title("NeoJaundice Regression: Test Predictions")
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_actual.png", dpi=160)
    plt.close()

    residuals = predicted - actual
    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residuals, alpha=0.65)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted target")
    plt.ylabel("Residual (prediction - actual)")
    plt.title("NeoJaundice Regression: Test Residuals")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_plot.png", dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be >= 0.")
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_csv = args.manifest_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv}")
    split_metadata = pd.read_csv(test_csv)
    if "patient_id" not in split_metadata.columns:
        raise ValueError("test.csv must contain patient_id to support patient-level evaluation.")

    dataset = load_test_dataset(args.manifest_dir, args.image_root)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    model = load_model(args.checkpoint, device)

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images).reshape(-1)
            targets = batch["target"].reshape(-1)
            if outputs.shape[0] != targets.shape[0]:
                raise RuntimeError("Prediction and target batch lengths do not match.")
            patient_ids = batch["patient_id"]
            image_ids = batch.get("image_id", [""] * len(patient_ids))
            for pid, image_id, target, prediction in zip(
                patient_ids,
                image_ids,
                targets.detach().cpu().numpy(),
                outputs.detach().cpu().numpy(),
            ):
                rows.append({
                    "patient_id": str(pid),
                    "image_id": str(image_id),
                    "target": float(target),
                    "prediction": float(prediction),
                    "residual": float(prediction - target),
                    "absolute_error": float(abs(prediction - target)),
                })

    if not rows:
        raise RuntimeError("The test dataloader returned zero predictions.")

    predictions = pd.DataFrame(rows)
    patient_predictions = patient_aggregate(predictions)

    image_metrics = calculate_regression_metrics(
        predictions["target"].to_numpy(), predictions["prediction"].to_numpy()
    )
    patient_metrics = calculate_regression_metrics(
        patient_predictions["target"].to_numpy(),
        patient_predictions["prediction"].to_numpy(),
    )

    # Training-set mean is a simple baseline. It is derived only from train.csv.
    train_csv = args.manifest_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_csv}")
    train_frame = pd.read_csv(train_csv)
    if args.target_column not in train_frame.columns:
        raise ValueError(
            f"Target column {args.target_column!r} not found in train.csv. "
            f"Available columns: {train_frame.columns.tolist()}"
        )
    train_targets = pd.to_numeric(train_frame[args.target_column], errors="coerce").dropna()
    if train_targets.empty:
        raise ValueError("No numeric target values found in train.csv.")
    baseline_value = float(train_targets.mean())
    baseline_patient_metrics = calculate_regression_metrics(
        patient_predictions["target"].to_numpy(),
        np.full(len(patient_predictions), baseline_value, dtype=float),
    )

    ci = bootstrap_patient_mae(
        patient_predictions, args.bootstrap_replicates, args.seed
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)

    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    patient_predictions.to_csv(args.output_dir / "patient_predictions.csv", index=False)
    make_plots(predictions, args.output_dir)

    result = {
        "task": "regression",
        "evaluation_split": "test",
        "test_set_used_for_model_selection": False,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "seed": args.seed,
        "preprocessing": "resize",
        "target_column": args.target_column,
        "target_measurement_semantics": (
            "Not independently verified; interpret target units according to source dataset documentation."
        ),
        "image_level": image_metrics,
        "patient_level": patient_metrics,
        "training_mean_baseline_patient_level": {
            "constant_prediction": baseline_value,
            **baseline_patient_metrics,
        },
        "patient_level_mae_bootstrap_ci": ci,
        "notes": [
            "No training or hyperparameter tuning was performed by this script.",
            "Patient-level metrics aggregate predictions by patient_id.",
            "Bootstrap resampling is performed at the patient level.",
            "This research evaluation is not clinical validation.",
        ],
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    report = f"""# NeoJaundice Regression Test Evaluation

## Model identity
- Checkpoint: `{args.checkpoint.resolve()}`
- SHA-256: `{checkpoint_sha256}`
- Device: `{device}`
- Preprocessing: `resize`
- Test set used for model selection: **No**

## Image-level metrics
- Images: {image_metrics['count']}
- MAE: {image_metrics['mae']:.4f} target units
- RMSE: {image_metrics['rmse']:.4f} target units
- Prediction bias (prediction − target): {image_metrics['bias_prediction_minus_target']:.4f}

## Patient-level metrics
- Patients: {patient_metrics['count']}
- MAE: {patient_metrics['mae']:.4f} target units
- RMSE: {patient_metrics['rmse']:.4f} target units
- Prediction bias (prediction − target): {patient_metrics['bias_prediction_minus_target']:.4f}

## Training-mean baseline
- Constant prediction: {baseline_value:.4f}
- Patient-level MAE: {baseline_patient_metrics['mae']:.4f}
- Patient-level RMSE: {baseline_patient_metrics['rmse']:.4f}

## Patient-level bootstrap interval
{(
    f"- 95% bootstrap CI for patient-level MAE: "
    f"[{ci['patient_mae_ci_lower']:.4f}, {ci['patient_mae_ci_upper']:.4f}] "
    f"({ci['replicates']} replicates)"
    if ci else "- Not calculated."
)}

## Limitations
- The target measurement semantics and units must be verified against the original dataset documentation.
- Results describe performance on this dataset split and do not establish clinical validity.
- Do not use these results to make treatment decisions.
"""
    (args.output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")

    print("\nRegression test evaluation completed.")
    print(json.dumps({
        "image_level": image_metrics,
        "patient_level": patient_metrics,
        "baseline_patient_level": baseline_patient_metrics,
        "checkpoint_sha256": checkpoint_sha256,
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

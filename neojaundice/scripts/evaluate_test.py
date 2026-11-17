"""Evaluate a frozen NeoJaundice regression checkpoint on the held-out test split.

No training or model/threshold tuning occurs here. Patient is the primary
statistical unit; image-level results are reported as secondary analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/neojaundice")
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/external/neojaundice/extracted/NeoJaundice"),
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluation/regression_final")
    )
    p.add_argument(
        "--preprocessing", choices=("resize", "letterbox", "augment"), default="augment"
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-column", default="bilirubin_value")
    p.add_argument(
        "--target-unit",
        default="unverified",
        help="Set to mg/dL only after verifying source documentation.",
    )
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path, device: torch.device):
    from ml.training.model import build_model

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Only for older PyTorch versions that do not support weights_only.
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint must be a dictionary.")
    state = next(
        (
            obj[k]
            for k in ("model_state_dict", "state_dict", "model")
            if isinstance(obj.get(k), dict)
            and obj[k]
            and all(isinstance(v, torch.Tensor) for v in obj[k].values())
        ),
        None,
    )
    if state is None and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        state = obj
    if state is None:
        raise ValueError("No supported model state_dict found in checkpoint.")
    model = build_model(pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def patient_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("patient_id")["target"].nunique()
    bad = counts[counts > 1]
    if not bad.empty:
        raise ValueError(
            f"Inconsistent regression targets for patients: {bad.index.astype(str).tolist()[:20]}"
        )
    return df.groupby("patient_id", as_index=False).agg(
        target=("target", "first"),
        prediction=("prediction", "mean"),
        image_count=("prediction", "size"),
    )


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    abs_err = np.abs(err)
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty array.")
    return {
        "count": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "median_absolute_error": float(np.median(abs_err)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 and np.var(y) > 0 else None,
        "mean_target": float(np.mean(y)),
        "mean_prediction": float(np.mean(pred)),
        "bias_prediction_minus_target": float(np.mean(err)),
        "within_1_target_unit_pct": float(np.mean(abs_err <= 1.0) * 100),
        "within_2_target_units_pct": float(np.mean(abs_err <= 2.0) * 100),
    }


def bootstrap_mae(patient_df: pd.DataFrame, n: int, seed: int):
    if n <= 0:
        return None
    rng = np.random.default_rng(seed)
    y = patient_df.target.to_numpy(float)
    p = patient_df.prediction.to_numpy(float)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        vals[i] = np.mean(np.abs(p[idx] - y[idx]))
    return {
        "replicates": int(n),
        "confidence_level": 0.95,
        "mae_ci_lower": float(np.quantile(vals, 0.025)),
        "mae_ci_upper": float(np.quantile(vals, 0.975)),
    }


def save_regression_plots(df: pd.DataFrame, out: Path, unit: str) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    y = df.target.to_numpy(float)
    pred = df.prediction.to_numpy(float)
    residual = pred - y
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    pad = max((hi - lo) * 0.04, 0.1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, pred, alpha=0.65, edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", label="Identity")
    ax.set(
        xlabel=f"Reference ({unit})",
        ylabel=f"Prediction ({unit})",
        title="Test set: predicted vs reference",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "predicted_vs_actual.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pred, residual, alpha=0.65, edgecolors="none")
    ax.axhline(0, linestyle="--")
    ax.set(
        xlabel=f"Predicted ({unit})",
        ylabel=f"Residual ({unit})",
        title="Test set: residuals vs prediction",
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residual, bins="auto", edgecolor="black", alpha=0.8)
    ax.axvline(0, linestyle="--")
    ax.set(
        xlabel=f"Residual ({unit})", ylabel="Image count", title="Residual distribution"
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_distribution.png", dpi=180)
    plt.close(fig)

    mean_pair = (y + pred) / 2
    bias = float(np.mean(residual))
    sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    lower, upper = bias - 1.96 * sd, bias + 1.96 * sd
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_pair, residual, alpha=0.65, edgecolors="none")
    ax.axhline(bias, linestyle="-", label=f"Bias = {bias:.3f}")
    ax.axhline(lower, linestyle="--", label=f"Lower LoA = {lower:.3f}")
    ax.axhline(upper, linestyle="--", label=f"Upper LoA = {upper:.3f}")
    ax.set(
        xlabel=f"Mean of reference and prediction ({unit})",
        ylabel=f"Prediction − reference ({unit})",
        title="Bland–Altman agreement plot",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "bland_altman.png", dpi=180)
    plt.close(fig)

    # Descriptive bins; do not interpret as clinical risk bands.
    bins = [-np.inf, 5, 10, 15, 20, np.inf]
    labels = ["≤5", "(5,10]", "(10,15]", "(15,20]", ">20"]
    ranges = pd.cut(y, bins=bins, labels=labels)
    grouped = pd.DataFrame(
        {"range": ranges, "absolute_error": np.abs(residual)}
    ).dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        grouped.loc[grouped.range == label, "absolute_error"].to_numpy()
        for label in labels
    ]
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set(
        xlabel="Reference target range (dataset units)",
        ylabel="Absolute error (dataset units)",
        title="Absolute error by reference-value range",
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "error_by_target_range.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    tolerances = [0.5, 1, 1.5, 2, 3, 4, 5]
    percentages = [np.mean(np.abs(residual) <= t) * 100 for t in tolerances]
    ax.plot(tolerances, percentages, marker="o")
    ax.set(
        xlabel="Absolute-error tolerance (dataset units)",
        ylabel="Images within tolerance (%)",
        ylim=(0, 100),
        title="Error tolerance profile",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "error_tolerance.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.bootstrap_replicates < 0:
        raise ValueError(
            "batch-size must be positive; workers/bootstrap replicates cannot be negative."
        )
    set_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    manifest = args.manifest_dir / "test.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    from ml.training.dataset import NeoJaundiceDataset

    dataset = NeoJaundiceDataset(
        manifest=manifest,
        image_root=args.image_root,
        task="regression",
        preprocessing=args.preprocessing,
        training=False,
        exclude_zero=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = load_checkpoint(args.checkpoint, device)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device)).reshape(-1).cpu().numpy()
            target = batch["target"].reshape(-1).cpu().numpy()
            if len(output) != len(target):
                raise RuntimeError("Prediction/target batch size mismatch.")
            for pid, iid, y, pred in zip(
                batch["patient_id"], batch["image_id"], target, output
            ):
                rows.append(
                    {
                        "patient_id": str(pid),
                        "image_id": str(iid),
                        "target": float(y),
                        "prediction": float(pred),
                        "residual": float(pred - y),
                        "absolute_error": float(abs(pred - y)),
                    }
                )
    if not rows:
        raise RuntimeError("No test predictions were produced.")
    image_df = pd.DataFrame(rows)
    patient_df = patient_aggregate(image_df)
    image_metrics = regression_metrics(
        image_df.target.to_numpy(), image_df.prediction.to_numpy()
    )
    patient_metrics = regression_metrics(
        patient_df.target.to_numpy(), patient_df.prediction.to_numpy()
    )

    train_csv = args.manifest_dir / "train.csv"
    train = pd.read_csv(train_csv)
    if args.target_column not in train:
        raise ValueError(f"{args.target_column!r} missing from {train_csv}")
    train_values = pd.to_numeric(train[args.target_column], errors="coerce")
    train_pairs = pd.DataFrame(
        {"patient_id": train["patient_id"], "target": train_values}
    ).dropna()
    train_consistency = train_pairs.groupby("patient_id").target.nunique()
    if (train_consistency > 1).any():
        raise ValueError(
            "Training target differs across images for one or more patients; baseline not calculated."
        )
    baseline_value = float(train_pairs.drop_duplicates("patient_id").target.mean())
    baseline_metrics = regression_metrics(
        patient_df.target.to_numpy(), np.full(len(patient_df), baseline_value)
    )
    ci = bootstrap_mae(patient_df, args.bootstrap_replicates, args.seed)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    image_df.to_csv(out / "test_predictions.csv", index=False)
    patient_df.to_csv(out / "patient_predictions.csv", index=False)
    save_regression_plots(image_df, out, args.target_unit)
    digest = sha256_file(args.checkpoint)
    result = {
        "task": "regression",
        "evaluation_split": "test",
        "test_set_used_for_model_selection": False,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "device": str(device),
        "seed": args.seed,
        "preprocessing": args.preprocessing,
        "target_column": args.target_column,
        "target_unit": args.target_unit,
        "image_level": image_metrics,
        "patient_level": patient_metrics,
        "training_patient_mean_baseline": {
            "constant_prediction": baseline_value,
            **baseline_metrics,
        },
        "patient_level_mae_bootstrap_ci": ci,
        "notes": [
            "Patient-level bootstrap resamples patients, not images.",
            "Error tolerances are descriptive, not clinical acceptance criteria.",
            "Bland–Altman limits describe agreement in this sample only.",
            "Research evaluation is not clinical validation.",
        ],
    }
    (out / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = f"""# NeoJaundice Regression Test Evaluation

## Model identity
- Checkpoint: `{args.checkpoint.resolve()}`
- SHA-256: `{digest}`
- Device: `{device}`
- Preprocessing: `{args.preprocessing}` (deterministic inference; training augmentations disabled)
- Target column: `{args.target_column}`
- Target unit: `{args.target_unit}`
- Test set used for model selection: **No**

## Image-level metrics
- Images: {image_metrics["count"]}
- MAE: {image_metrics["mae"]:.4f}
- Median absolute error: {image_metrics["median_absolute_error"]:.4f}
- RMSE: {image_metrics["rmse"]:.4f}
- R²: {image_metrics["r2"]}
- Bias (prediction − reference): {image_metrics["bias_prediction_minus_target"]:.4f}
- Within ±1 target unit: {image_metrics["within_1_target_unit_pct"]:.2f}%
- Within ±2 target units: {image_metrics["within_2_target_units_pct"]:.2f}%

## Patient-level metrics
- Patients: {patient_metrics["count"]}
- MAE: {patient_metrics["mae"]:.4f}
- Median absolute error: {patient_metrics["median_absolute_error"]:.4f}
- RMSE: {patient_metrics["rmse"]:.4f}
- R²: {patient_metrics["r2"]}
- Bias: {patient_metrics["bias_prediction_minus_target"]:.4f}
- Within ±1 target unit: {patient_metrics["within_1_target_unit_pct"]:.2f}%
- Within ±2 target units: {patient_metrics["within_2_target_units_pct"]:.2f}%

## Training-patient-mean baseline
- Constant prediction: {baseline_value:.4f}
- Patient-level MAE: {baseline_metrics["mae"]:.4f}
- Patient-level RMSE: {baseline_metrics["rmse"]:.4f}

## Patient-level bootstrap interval
{f"- 95% CI for patient-level MAE: [{ci['mae_ci_lower']:.4f}, {ci['mae_ci_upper']:.4f}] ({ci['replicates']} replicates)" if ci else "- Not calculated."}

## Plots
See `plots/` for predicted-vs-reference, residual, residual-distribution, Bland–Altman, error-by-range and error-tolerance plots.

## Limitations
- Confirm target semantics and units from original dataset documentation.
- Error bands are descriptive and are not clinical thresholds.
- Results on this split do not establish clinical validity or treatment safety.
"""
    (out / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "image_level": image_metrics,
                "patient_level": patient_metrics,
                "baseline_patient_level": baseline_metrics,
                "bootstrap_ci": ci,
                "checkpoint_sha256": digest,
                "output_dir": str(out.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate patient-level validation probabilities and select a frozen classifier threshold.

Default selection is Youden's J (sensitivity + specificity - 1), a generic
statistical criterion. It is NOT a clinically justified screening operating
point. A clinical sensitivity target must be agreed with domain experts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=".")
    p.add_argument(
        "--validation-manifest", default="data/processed/neojaundice/val.csv"
    )
    p.add_argument("--image-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--output-dir", default="artifacts/evaluation/classification_validation"
    )
    p.add_argument(
        "--preprocessing", choices=("resize", "letterbox", "augment"), default="augment"
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection", choices=("youden", "f1"), default="youden")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_model(checkpoint, device):
    from ml.training.model import build_model

    try:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("Invalid checkpoint wrapper.")
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
        raise ValueError("No model state_dict found.")
    if all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    model = build_model(pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    a = args_parser()
    root = Path(a.project_root).resolve()
    val_path = Path(a.validation_manifest)
    val_path = val_path if val_path.is_absolute() else root / val_path
    image_root = Path(a.image_root)
    image_root = image_root if image_root.is_absolute() else root / image_root
    ckpt = Path(a.checkpoint)
    ckpt = ckpt if ckpt.is_absolute() else root / ckpt
    out = Path(a.output_dir)
    out = out if out.is_absolute() else root / out
    for p in (val_path, image_root, ckpt):
        if not p.exists():
            raise FileNotFoundError(p)
    frame = pd.read_csv(val_path)
    required = {"patient_id", "image_id", "treatment_label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Validation CSV missing {sorted(missing)}")
    labels = pd.to_numeric(frame.treatment_label, errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1.")
    frame["treatment_label"] = labels.astype(int)
    consistency = frame.groupby("patient_id").treatment_label.nunique()
    if (consistency > 1).any():
        raise ValueError("Inconsistent labels within validation patients.")
    import sys

    sys.path.insert(0, str(root))
    from ml.training.dataset import NeoJaundiceDataset

    dataset = NeoJaundiceDataset(
        frame,
        image_root=str(image_root),
        task="classification",
        preprocessing=a.preprocessing,
        training=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=device.type == "cuda",
    )
    rows = []
    with torch.inference_mode():
        for batch in loader:
            probs = (
                torch.sigmoid(model(batch["image"].to(device)).reshape(-1))
                .cpu()
                .numpy()
            )
            targets = torch.as_tensor(batch["target"]).reshape(-1).cpu().numpy()
            for pid, iid, y, p in zip(
                batch["patient_id"], batch["image_id"], targets, probs
            ):
                rows.append(
                    {
                        "patient_id": str(pid),
                        "image_id": str(iid),
                        "target": int(y),
                        "probability": float(p),
                    }
                )
    image = pd.DataFrame(rows)
    if len(image) != len(frame):
        raise RuntimeError("Validation inference count differs from manifest.")
    patient = image.groupby("patient_id", as_index=False).agg(
        target=("target", "first"),
        probability=("probability", "mean"),
        image_count=("image_id", "size"),
    )
    if patient.target.nunique() != 2:
        raise ValueError("Both classes are required for threshold selection.")
    thresholds = np.unique(
        np.concatenate([np.linspace(0.01, 0.99, 99), patient.probability.to_numpy()])
    )
    records = []
    y = patient.target.to_numpy(int)
    prob = patient.probability.to_numpy(float)
    for t in thresholds:
        pred = (prob >= t).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        prec = precision_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        records.append(
            {
                "threshold": float(t),
                "sensitivity": sens,
                "specificity": spec,
                "precision": float(prec),
                "f1": float(f1),
                "youden_j": float(sens + spec - 1),
                "accuracy": float(accuracy_score(y, pred)),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    table = pd.DataFrame(records)
    score_col = "youden_j" if a.selection == "youden" else "f1"
    # Tie-break: choose the higher threshold, which generally avoids extra positives.
    selected = table.sort_values(
        [score_col, "threshold"], ascending=[False, False]
    ).iloc[0]
    threshold = float(selected.threshold)
    out.mkdir(parents=True, exist_ok=True)
    image.to_csv(out / "validation_image_predictions.csv", index=False)
    patient.to_csv(out / "validation_patient_predictions.csv", index=False)
    table.to_csv(out / "threshold_analysis.csv", index=False)
    threshold_record = {
        "threshold": threshold,
        "selection_method": a.selection,
        "selection_unit": "patient",
        "selection_score": score_col,
        "selection_score_value": float(selected[score_col]),
        "tie_break": "higher threshold",
        "validation_patient_count": int(len(patient)),
        "checkpoint_sha256": sha256(ckpt),
        "warning": "Exploratory statistical threshold only; not a clinically validated screening threshold.",
    }
    (out / "selected_threshold.json").write_text(
        json.dumps(threshold_record, indent=2) + "\n", encoding="utf-8"
    )
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    fpr, tpr, _ = roc_curve(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, prob):.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(
        xlabel="False positive rate",
        ylabel="Sensitivity",
        title="Validation ROC — patient level",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "roc_curve.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"AP={average_precision_score(y, prob):.3f}")
    ax.set(
        xlabel="Recall / sensitivity",
        ylabel="Precision",
        title="Validation precision–recall — patient level",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "precision_recall_curve.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ("sensitivity", "specificity", "precision", "f1"):
        ax.plot(table.threshold, table[col], label=col)
    ax.axvline(threshold, linestyle="--", label=f"selected={threshold:.3f}")
    ax.set(
        xlabel="Threshold",
        ylabel="Metric",
        ylim=(0, 1.02),
        title="Validation threshold analysis",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots / "threshold_metrics.png", dpi=180)
    plt.close(fig)
    frac, mean = calibration_curve(y, prob, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean, frac, marker="o")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Validation calibration — patient level",
    )
    fig.tight_layout()
    fig.savefig(plots / "calibration_curve.png", dpi=180)
    plt.close(fig)
    summary = {
        "image_count": len(image),
        "patient_count": len(patient),
        "patient_roc_auc": float(roc_auc_score(y, prob)),
        "patient_average_precision": float(average_precision_score(y, prob)),
        "selected_threshold": threshold_record,
    }
    (out / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

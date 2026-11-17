"""Evaluate a frozen NeoJaundice classifier on test data using a preselected threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def binary_metrics(y_true, probabilities, threshold):
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels/probabilities must have equal nonzero lengths.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must be binary.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite in [0,1].")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "count": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p))
        if len(np.unique(y)) == 2
        else None,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def bootstrap_patient_ci(df, n, seed, threshold):
    if n <= 0:
        return {"requested_replicates": 0, "note": "Bootstrap disabled."}
    rng = np.random.default_rng(seed)
    rows = df.reset_index(drop=True)
    metrics = {
        k: []
        for k in (
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1",
            "roc_auc",
        )
    }
    for _ in range(n):
        sample = rows.iloc[rng.integers(0, len(rows), size=len(rows))]
        y = sample.target.to_numpy(int)
        p = sample.probability.to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        m = binary_metrics(y, p, threshold)
        for k in metrics:
            if m[k] is not None and np.isfinite(m[k]):
                metrics[k].append(m[k])
    return {
        "method": f"patient-level percentile bootstrap; fixed threshold={threshold:.6f}",
        "requested_replicates": n,
        "valid_replicates": min((len(v) for v in metrics.values()), default=0),
        "seed": seed,
        "intervals": {
            k: (
                {
                    "lower_95": float(np.percentile(v, 2.5)),
                    "upper_95": float(np.percentile(v, 97.5)),
                    "valid_replicates": len(v),
                }
                if v
                else None
            )
            for k, v in metrics.items()
        },
    }


def load_model(path, device):
    from ml.training.model import build_model

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("Invalid checkpoint.")
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
        raise ValueError("No state_dict found.")
    if all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    model = build_model(pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def save_plots(patient, threshold, out):
    plotdir = out / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    y = patient.target.to_numpy(int)
    p = patient.probability.to_numpy(float)
    pred = (p >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(
        xlabel="False positive rate",
        ylabel="Sensitivity",
        title="Test ROC — patient level",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / "roc_curve.png", dpi=180)
    plt.close(fig)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"AP={average_precision_score(y, p):.3f}")
    ax.set(
        xlabel="Recall / sensitivity",
        ylabel="Precision",
        title="Test precision–recall — patient level",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / "precision_recall_curve.png", dpi=180)
    plt.close(fig)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_title(f"Test confusion matrix (threshold={threshold:.3f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plotdir / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    frac, mean = calibration_curve(y, p, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean, frac, marker="o")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Test calibration — patient level",
    )
    fig.tight_layout()
    fig.savefig(plotdir / "calibration_curve.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=".")
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--threshold-file",
        required=True,
        help="selected_threshold.json created from validation only",
    )
    p.add_argument("--output-dir", default="artifacts/evaluation/classification_final")
    p.add_argument(
        "--preprocessing", choices=("resize", "letterbox", "augment"), default="augment"
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root = Path(a.project_root).resolve()

    def resolve(v):
        x = Path(v).expanduser()
        return x if x.is_absolute() else root / x

    manifest, images, ckpt, threshold_file, out = map(
        resolve,
        (a.test_manifest, a.image_root, a.checkpoint, a.threshold_file, a.output_dir),
    )
    for x in (manifest, images, ckpt, threshold_file):
        if not x.exists():
            raise FileNotFoundError(x)
    selected = json.loads(threshold_file.read_text(encoding="utf-8"))
    threshold = float(selected["threshold"])
    if selected.get("checkpoint_sha256") != sha256(ckpt):
        raise ValueError(
            "Threshold file checkpoint SHA-256 does not match selected classifier."
        )
    if selected.get("selection_unit") != "patient":
        raise ValueError("Threshold must have been selected at patient level.")
    if not 0 < threshold < 1:
        raise ValueError("Invalid threshold in threshold file.")
    frame = pd.read_csv(manifest)
    required = {"patient_id", "image_id", "treatment_label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Test CSV missing {sorted(missing)}")
    labels = pd.to_numeric(frame.treatment_label, errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("treatment_label must be binary 0/1.")
    frame["treatment_label"] = labels.astype(int)
    consistency = frame.groupby("patient_id").treatment_label.nunique()
    if (consistency > 1).any():
        raise ValueError("Inconsistent treatment labels within test patients.")
    sys.path.insert(0, str(root))
    from ml.training.dataset import NeoJaundiceDataset

    dataset = NeoJaundiceDataset(
        frame,
        image_root=str(images),
        task="classification",
        preprocessing=a.preprocessing,
        training=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, device)
    loader = DataLoader(
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
            for pid, iid, y, prob in zip(
                batch["patient_id"], batch["image_id"], targets, probs
            ):
                rows.append(
                    {
                        "patient_id": str(pid),
                        "image_id": str(iid),
                        "target": int(y),
                        "probability": float(prob),
                    }
                )
    image = pd.DataFrame(rows)
    if len(image) != len(frame):
        raise RuntimeError("Prediction count differs from test manifest.")
    patient = image.groupby("patient_id", as_index=False).agg(
        target=("target", "first"),
        probability=("probability", "mean"),
        image_count=("image_id", "size"),
    )
    if patient.target.nunique() != 2:
        raise ValueError("Test set must contain both classes for ROC/PR analysis.")
    image["prediction"] = (image.probability >= threshold).astype(int)
    patient["prediction"] = (patient.probability >= threshold).astype(int)
    im = binary_metrics(image.target, image.probability, threshold)
    pm = binary_metrics(patient.target, patient.probability, threshold)
    ci = bootstrap_patient_ci(patient, a.bootstrap_replicates, a.seed, threshold)
    out.mkdir(parents=True, exist_ok=True)
    image.to_csv(out / "test_predictions.csv", index=False)
    patient.to_csv(out / "patient_predictions.csv", index=False)
    save_plots(patient, threshold, out)
    report = {
        "task": "binary dataset-label classification",
        "evaluation_split": "test",
        "target_column": "treatment_label",
        "target_semantics": "Unverified; not a clinical diagnosis or treatment recommendation.",
        "test_manifest": str(manifest),
        "image_count": len(image),
        "patient_count": len(patient),
        "threshold": threshold,
        "threshold_policy": selected,
        "device": str(device),
        "preprocessing": a.preprocessing,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "image_level": im,
        "patient_level": pm,
        "patient_level_bootstrap_95ci": ci,
        "classification_report_patient_level": classification_report(
            patient.target,
            patient.prediction,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
        "limitations": [
            "Threshold selected using validation patients only; test data did not select threshold.",
            "Threshold selection criterion is statistical and not clinically validated.",
            "Dataset target semantics are unverified; results do not establish clinical screening performance.",
        ],
    }
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    md = f"""# NeoJaundice Classification Test Evaluation

> Research-only. `treatment_label` semantics are unverified; do not interpret this as clinical screening accuracy.

- Checkpoint SHA-256: `{report["checkpoint_sha256"]}`
- Threshold: `{threshold:.6f}` (selected using validation patients; method: {selected.get("selection_method")})
- Preprocessing: `{a.preprocessing}`
- Images: {len(image)}
- Patients: {len(patient)}

## Patient-level metrics

```json
{json.dumps(pm, indent=2)}
```

## Image-level metrics

```json
{json.dumps(im, indent=2)}
```

## Patient-level bootstrap confidence intervals

```json
{json.dumps(ci, indent=2)}
```

Plots are in `plots/`: ROC, precision–recall, confusion matrix and calibration.
The threshold is an exploratory statistical operating point, not a clinically approved screening threshold.
"""
    (out / "evaluation_report.md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "threshold": threshold,
                "image_level": im,
                "patient_level": pm,
                "bootstrap": ci,
                "output_dir": str(out.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

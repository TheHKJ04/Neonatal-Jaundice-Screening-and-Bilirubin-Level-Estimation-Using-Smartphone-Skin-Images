"""Image-level and patient-aggregated exploratory metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def summarize_predictions(records: list[dict], task: str) -> dict:
    if not records:
        raise ValueError("No predictions to evaluate")
    frame = pd.DataFrame(records)
    result = {}
    for level, data in (
        ("image", frame),
        (
            "patient",
            frame.groupby("patient_id", as_index=False).agg(
                target=("target", "mean"), prediction=("prediction", "mean")
            ),
        ),
    ):
        y = data["target"].to_numpy(dtype=float)
        pred = data["prediction"].to_numpy(dtype=float)
        if task == "regression":
            result[level] = {
                "count": len(data),
                "mae": float(mean_absolute_error(y, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            }
        elif task == "classification":
            if level == "patient":
                # All image labels should be identical within each patient.
                original = frame.groupby("patient_id")["target"].nunique()
                if (original != 1).any():
                    raise ValueError("Conflicting patient classification labels")
            labels = (pred >= 0.5).astype(int)
            result[level] = {
                "count": len(data),
                "accuracy": float(accuracy_score(y, labels)),
                "f1": float(f1_score(y, labels, zero_division=0)),
                "roc_auc": float(roc_auc_score(y, pred))
                if len(np.unique(y)) == 2
                else None,
            }
        else:
            raise ValueError(f"Unknown task: {task}")
    return result

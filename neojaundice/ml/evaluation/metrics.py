from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    y_pred,
    y_score=None,
    positive_label="1",
):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    if len(y_true) == 0:
        raise ValueError("Empty classification dataset.")

    if len(y_true) != len(y_pred):
        raise ValueError("Prediction and label lengths differ.")

    # Predictions may legitimately contain a class absent
    # from the ground-truth test observations.
    labels = sorted(set(y_true) | set(y_pred))

    binary = len(labels) == 2

    result = {
        "samples": int(len(y_true)),
        "classes": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    if binary:
        positive = str(positive_label)

        if positive not in labels:
            raise ValueError("Positive label not found.")

        negative = next(x for x in labels if x != positive)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[negative, positive],
        ).ravel()

        result.update(
            {
                "positive_precision": float(
                    precision_score(
                        y_true,
                        y_pred,
                        pos_label=positive,
                        zero_division=0,
                    )
                ),
                "positive_recall": float(
                    recall_score(
                        y_true,
                        y_pred,
                        pos_label=positive,
                        zero_division=0,
                    )
                ),
                "positive_f1": float(
                    f1_score(
                        y_true,
                        y_pred,
                        pos_label=positive,
                        zero_division=0,
                    )
                ),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
                "specificity": float(tn / (tn + fp)) if tn + fp else None,
                "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
            }
        )

    if y_score is not None:
        scores = np.asarray(y_score, dtype=float)

        if not np.isfinite(scores).all():
            raise ValueError("Invalid probability values.")

        if binary:
            if scores.ndim != 1:
                raise ValueError(
                    "Binary classification requires one positive-class "
                    "probability per sample."
                )

            if np.any((scores < 0) | (scores > 1)):
                raise ValueError("Probabilities must be between 0 and 1.")

            y_binary = (y_true == str(positive_label)).astype(int)

            if len(np.unique(y_binary)) == 2:
                result["roc_auc"] = float(roc_auc_score(y_binary, scores))
                result["pr_auc_average_precision"] = float(
                    average_precision_score(y_binary, scores)
                )

        elif scores.ndim == 2:
            if scores.shape != (len(y_true), len(labels)):
                raise ValueError("Incorrect multiclass score dimensions.")

            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    scores,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )
            )

    return result


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) == 0:
        raise ValueError("Empty regression dataset.")

    if len(y_true) != len(y_pred):
        raise ValueError("Prediction and label lengths differ.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Invalid regression values.")

    if np.any(y_true < 0):
        raise ValueError("Negative ground-truth bilirubin.")

    errors = y_pred - y_true

    result = {
        "samples": int(len(y_true)),
        "mae_mg_dl": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse_mg_dl": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "mean_bias_mg_dl": float(np.mean(errors)),
        "median_absolute_error_mg_dl": float(np.median(np.abs(errors))),
        "within_1_mg_dl": float(np.mean(np.abs(errors) <= 1)),
        "within_2_mg_dl": float(np.mean(np.abs(errors) <= 2)),
    }

    if len(y_true) >= 2:
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            result["pearson_correlation"] = float(pearsonr(y_true, y_pred).statistic)
            result["spearman_correlation"] = float(spearmanr(y_true, y_pred).statistic)
        else:
            result["pearson_correlation"] = None
            result["spearman_correlation"] = None

    return result


def cluster_bootstrap_mae_ci(
    y_true,
    y_pred,
    patient_ids,
    n_bootstrap=2000,
    seed=42,
):
    from sklearn.metrics import mean_absolute_error

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    patient_ids = np.asarray(patient_ids).astype(str)

    if not (len(y_true) == len(y_pred) == len(patient_ids)):
        raise ValueError("Input lengths differ.")

    if len(np.unique(patient_ids)) < 2:
        raise ValueError("At least two patients are required.")

    unique_ids = np.unique(patient_ids)
    rng = np.random.default_rng(seed)

    indices_by_patient = {pid: np.flatnonzero(patient_ids == pid) for pid in unique_ids}

    estimates = []

    for _ in range(n_bootstrap):
        sampled = rng.choice(
            unique_ids,
            size=len(unique_ids),
            replace=True,
        )

        indices = np.concatenate([indices_by_patient[pid] for pid in sampled])

        estimates.append(
            mean_absolute_error(
                y_true[indices],
                y_pred[indices],
            )
        )

    lower, upper = np.percentile(estimates, [2.5, 97.5])

    return {
        "mae_ci_95_lower": float(lower),
        "mae_ci_95_upper": float(upper),
        "unique_patients": int(len(unique_ids)),
        "bootstrap_iterations": int(n_bootstrap),
    }

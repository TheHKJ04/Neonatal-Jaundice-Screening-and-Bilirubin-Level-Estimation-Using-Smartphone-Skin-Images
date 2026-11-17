from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ml.evaluation.metrics import (
    classification_metrics,
    cluster_bootstrap_mae_ci,
    regression_metrics,
)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, allow_nan=False)


def evaluate_classification(args, output_dir, reports):
    if not args.classification_csv:
        return

    df = pd.read_csv(args.classification_csv)

    required = [args.class_true_col, args.class_pred_col]

    if args.score_col:
        required.append(args.score_col)

    if args.score_cols:
        required.extend(args.score_cols)

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing classification columns: {missing}")

    df = df.dropna(subset=required).copy()

    scores = None

    if args.score_col:
        scores = pd.to_numeric(df[args.score_col]).to_numpy()

    elif args.score_cols:
        scores = df[args.score_cols].apply(pd.to_numeric, errors="raise").to_numpy()

    result = classification_metrics(
        y_true=df[args.class_true_col],
        y_pred=df[args.class_pred_col],
        y_score=scores,
        positive_label=args.positive_label,
    )

    save_json(result, output_dir / "classification_metrics.json")

    reports.update(
        {
            f"classification_{key}": value
            for key, value in result.items()
            if isinstance(value, (int, float)) or value is None
        }
    )

    labels = result["classes"]

    cm = confusion_matrix(
        df[args.class_true_col].astype(str),
        df[args.class_pred_col].astype(str),
        labels=labels,
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print("\nCLASSIFICATION RESULTS")
    print(json.dumps(result, indent=4))


def evaluate_regression(args, output_dir, reports):
    if not args.regression_csv:
        return

    df = pd.read_csv(args.regression_csv)

    required = [args.reg_true_col, args.reg_pred_col]

    if args.patient_col:
        required.append(args.patient_col)

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing regression columns: {missing}")

    df[args.reg_true_col] = pd.to_numeric(df[args.reg_true_col], errors="coerce")
    df[args.reg_pred_col] = pd.to_numeric(df[args.reg_pred_col], errors="coerce")

    df = df.dropna(subset=required).copy()

    y_true = df[args.reg_true_col].to_numpy()
    y_pred = df[args.reg_pred_col].to_numpy()

    result = regression_metrics(y_true, y_pred)

    if args.patient_col:
        ci = cluster_bootstrap_mae_ci(
            y_true,
            y_pred,
            df[args.patient_col].to_numpy(),
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        result.update(ci)

    save_json(result, output_dir / "regression_metrics.json")

    reports.update(
        {
            f"regression_{key}": value
            for key, value in result.items()
            if isinstance(value, (int, float)) or value is None
        }
    )

    errors = y_pred - y_true
    averages = (y_true + y_pred) / 2

    # Actual versus predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    lower = min(y_true.min(), y_pred.min())
    upper = max(y_true.max(), y_pred.max())

    plt.plot([lower, upper], [lower, upper], "r--")
    plt.xlabel("Actual bilirubin (mg/dL)")
    plt.ylabel("Predicted bilirubin (mg/dL)")
    plt.title("Actual vs Predicted Bilirubin")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=200)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(7, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Prediction Error (mg/dL)")
    plt.title("Regression Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution.png", dpi=200)
    plt.close()

    # Bland-Altman plot
    bias = np.mean(errors)
    sd = np.std(errors, ddof=1) if len(errors) >= 2 else 0

    plt.figure(figsize=(7, 6))
    plt.scatter(averages, errors, alpha=0.6)
    plt.axhline(bias, color="red", linestyle="--", label="Mean bias")
    plt.axhline(
        bias + 1.96 * sd,
        color="green",
        linestyle="--",
        label="Upper agreement limit",
    )
    plt.axhline(
        bias - 1.96 * sd,
        color="green",
        linestyle="--",
        label="Lower agreement limit",
    )
    plt.xlabel("Mean of Actual and Predicted (mg/dL)")
    plt.ylabel("Predicted - Actual (mg/dL)")
    plt.title("Bland-Altman Agreement Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bland_altman.png", dpi=200)
    plt.close()

    print("\nREGRESSION RESULTS")
    print(json.dumps(result, indent=4))


def main():
    parser = argparse.ArgumentParser(
        description="NeoJaundice classification and regression evaluation."
    )

    parser.add_argument("--classification-csv", type=Path)
    parser.add_argument("--regression-csv", type=Path)

    parser.add_argument("--class-true-col", default="actual_label")
    parser.add_argument("--class-pred-col", default="predicted_label")
    parser.add_argument("--score-col")
    parser.add_argument("--score-cols", nargs="+")
    parser.add_argument("--positive-label", default="1")

    parser.add_argument("--reg-true-col", default="actual_bilirubin")
    parser.add_argument("--reg-pred-col", default="predicted_bilirubin")

    parser.add_argument("--patient-col", default="patient_id")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/final"),
    )

    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.classification_csv and not args.regression_csv:
        parser.error("Provide at least one prediction CSV.")

    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap iterations.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}

    evaluate_classification(args, args.output_dir, reports)
    evaluate_regression(args, args.output_dir, reports)

    save_json(reports, args.output_dir / "summary.json")

    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in reports.items()]
    ).to_csv(args.output_dir / "summary.csv", index=False)

    print("\nEvaluation completed.")
    print(f"Reports saved at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

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

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, allow_nan=False)


def evaluate_classification(args, output):
    if not args.classification_csv:
        return None

    df = pd.read_csv(args.classification_csv)

    required = [
        args.class_true,
        args.class_pred,
    ]

    if args.score_col:
        required.append(args.score_col)

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing classification columns: {missing}")

    df = df.dropna(subset=required)

    scores = None

    if args.score_col:
        scores = pd.to_numeric(
            df[args.score_col],
            errors="raise",
        ).to_numpy()

    result = classification_metrics(
        df[args.class_true],
        df[args.class_pred],
        y_score=scores,
        positive_label=args.positive_label,
    )

    save_json(
        result,
        output / "classification_metrics.json",
    )

    labels = result["classes"]

    cm = confusion_matrix(
        df[args.class_true].astype(str),
        df[args.class_pred].astype(str),
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

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        output / "confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    return result


def evaluate_regression(args, output):
    if not args.regression_csv:
        return None

    df = pd.read_csv(args.regression_csv)

    required = [
        args.reg_true,
        args.reg_pred,
    ]

    if args.patient_col:
        required.append(args.patient_col)

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing regression columns: {missing}")

    df[args.reg_true] = pd.to_numeric(
        df[args.reg_true],
        errors="coerce",
    )

    df[args.reg_pred] = pd.to_numeric(
        df[args.reg_pred],
        errors="coerce",
    )

    df = df.dropna(subset=required)

    actual = df[args.reg_true].to_numpy()
    predicted = df[args.reg_pred].to_numpy()

    result = regression_metrics(actual, predicted)

    if args.patient_col:
        result.update(
            cluster_bootstrap_mae_ci(
                actual,
                predicted,
                df[args.patient_col].to_numpy(),
                n_bootstrap=args.bootstrap,
                seed=args.seed,
            )
        )

    save_json(
        result,
        output / "regression_metrics.json",
    )

    errors = predicted - actual
    means = (predicted + actual) / 2

    # Actual vs predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.6)

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    plt.plot(
        [lower, upper],
        [lower, upper],
        "r--",
    )

    plt.xlabel("Actual Bilirubin (mg/dL)")
    plt.ylabel("Predicted Bilirubin (mg/dL)")
    plt.title("Actual vs Predicted")

    plt.tight_layout()
    plt.savefig(output / "actual_vs_predicted.png", dpi=200)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(7, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(0, color="red", linestyle="--")

    plt.xlabel("Prediction Error (mg/dL)")
    plt.title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(output / "residual_distribution.png", dpi=200)
    plt.close()

    # Bland-Altman
    bias = np.mean(errors)
    sd = np.std(errors, ddof=1) if len(errors) > 1 else 0

    plt.figure(figsize=(7, 6))
    plt.scatter(means, errors, alpha=0.6)

    plt.axhline(bias, color="red", linestyle="--")
    plt.axhline(bias + 1.96 * sd, color="green", linestyle="--")
    plt.axhline(bias - 1.96 * sd, color="green", linestyle="--")

    plt.xlabel("Mean of Actual and Predicted")
    plt.ylabel("Predicted - Actual")
    plt.title("Bland-Altman Analysis")

    plt.tight_layout()
    plt.savefig(output / "bland_altman.png", dpi=200)
    plt.close()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--classification-csv", type=Path)
    parser.add_argument("--regression-csv", type=Path)

    parser.add_argument("--class-true", default="actual_label")
    parser.add_argument("--class-pred", default="predicted_label")
    parser.add_argument("--score-col")

    parser.add_argument("--positive-label", default="1")

    parser.add_argument("--reg-true", default="actual_bilirubin")
    parser.add_argument("--reg-pred", default="predicted_bilirubin")

    parser.add_argument("--patient-col", default="patient_id")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/final"),
    )

    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.classification_csv and not args.regression_csv:
        parser.error("Provide at least one prediction CSV.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    classification = evaluate_classification(args, args.output_dir)

    regression = evaluate_regression(args, args.output_dir)

    summary = {
        "classification": classification,
        "regression": regression,
        "seed": args.seed,
    }

    save_json(
        summary,
        args.output_dir / "summary.json",
    )

    rows = []

    for task, result in summary.items():
        if isinstance(result, dict):
            for metric, value in result.items():
                if isinstance(value, (int, float)) or value is None:
                    rows.append(
                        {
                            "task": task,
                            "metric": metric,
                            "value": value,
                        }
                    )

    pd.DataFrame(rows).to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )

    print("\nEVALUATION COMPLETED")
    print(json.dumps(summary, indent=4))
    print("Output:", args.output_dir.resolve())


if __name__ == "__main__":
    main()

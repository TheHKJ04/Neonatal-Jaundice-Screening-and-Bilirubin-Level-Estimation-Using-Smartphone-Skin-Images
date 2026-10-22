"""Unit tests for pure metric/aggregation helpers in evaluate_test.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_test.py"
SPEC = importlib.util.spec_from_file_location("evaluate_test", SCRIPT)
evaluate_test = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(evaluate_test)


def test_regression_metrics_perfect_predictions():
    result = evaluate_test.calculate_regression_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert result["count"] == 3
    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)


def test_regression_metrics_known_error():
    result = evaluate_test.calculate_regression_metrics(
        np.array([1.0, 2.0]),
        np.array([2.0, 4.0]),
    )
    assert result["mae"] == pytest.approx(1.5)
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))


def test_patient_aggregation_averages_images():
    frame = pd.DataFrame({
        "patient_id": ["A", "A", "B"],
        "target": [10.0, 10.0, 12.0],
        "prediction": [8.0, 12.0, 11.0],
    })
    result = evaluate_test.patient_aggregate(frame).sort_values("patient_id")
    assert result["patient_id"].tolist() == ["A", "B"]
    assert result["target"].tolist() == [10.0, 12.0]
    assert result["prediction"].tolist() == [10.0, 11.0]
    assert result["image_count"].tolist() == [2, 1]


def test_patient_aggregation_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        evaluate_test.patient_aggregate(pd.DataFrame({"patient_id": ["A"]}))


def test_patient_bootstrap_is_reproducible():
    frame = pd.DataFrame({
        "patient_id": ["A", "B", "C"],
        "target": [10.0, 12.0, 8.0],
        "prediction": [9.0, 14.0, 7.0],
    })
    first = evaluate_test.bootstrap_patient_mae(frame, 100, 42)
    second = evaluate_test.bootstrap_patient_mae(frame, 100, 42)
    assert first == second
    assert first["replicates"] == 100


def test_zero_bootstrap_skips_interval():
    frame = pd.DataFrame({
        "patient_id": ["A", "B"],
        "target": [10.0, 12.0],
        "prediction": [9.0, 14.0],
    })
    assert evaluate_test.bootstrap_patient_mae(frame, 0, 42) is None

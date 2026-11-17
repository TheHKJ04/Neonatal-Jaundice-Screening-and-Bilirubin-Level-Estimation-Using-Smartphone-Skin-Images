import pytest

from ml.evaluation.metrics import (
    classification_metrics,
    cluster_bootstrap_mae_ci,
    regression_metrics,
)


def test_classification():
    result = classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        y_score=[0.1, 0.7, 0.8, 0.9],
        positive_label="1",
    )

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_regression():
    result = regression_metrics(
        [10, 12, 15],
        [11, 11, 14],
    )

    assert result["mae_mg_dl"] == pytest.approx(1.0)
    assert result["rmse_mg_dl"] == pytest.approx(1.0)


def test_regression_invalid_values():
    with pytest.raises(ValueError):
        regression_metrics([10, -1], [9, 2])


def test_patient_bootstrap():
    result = cluster_bootstrap_mae_ci(
        [10, 12, 15, 20],
        [11, 11, 14, 19],
        ["A", "A", "B", "B"],
        n_bootstrap=200,
    )

    assert result["unique_patients"] == 2
    assert result["mae_ci_95_lower"] <= result["mae_ci_95_upper"]

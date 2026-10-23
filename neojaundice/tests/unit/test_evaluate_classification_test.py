import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "evaluate_classification_test.py"
spec = importlib.util.spec_from_file_location("evaluate_classification_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_binary_metrics_perfect_predictions():
    metrics = module.binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["recall_sensitivity"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_binary_metrics_single_class_auc_is_none():
    metrics = module.binary_metrics([0, 0], [0.1, 0.4], 0.5)
    assert metrics["roc_auc"] is None
    assert metrics["accuracy"] == 1.0


def test_binary_metrics_rejects_invalid_probability():
    try:
        module.binary_metrics([0, 1], [0.1, 1.2], 0.5)
    except ValueError as exc:
        assert "Probabilities" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid probability")


def test_sha256_is_deterministic(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"neojaundice")
    assert module.sha256_file(f) == module.sha256_file(f)

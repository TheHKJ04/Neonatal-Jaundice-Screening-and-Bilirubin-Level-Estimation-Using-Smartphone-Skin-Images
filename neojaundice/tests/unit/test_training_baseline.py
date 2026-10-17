"""Focused tests for model inputs, targets and patient aggregation."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from ml.training.dataset import NeoJaundiceDataset, make_transform
from ml.training.metrics import summarize_predictions
from ml.training.model import build_model, calculate_loss


def create_fixture(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 24), (120, 80, 50)).save(image_dir / "a.jpg")
    pd.DataFrame(
        [
            {
                "image_id": "a.jpg",
                "patient_id": "0001",
                "image_path": "images/a.jpg",
                "bilirubin_value": 8.5,
                "treatment_label": 1,
            }
        ]
    ).to_csv(tmp_path / "train.csv", index=False)
    return tmp_path / "train.csv"


@pytest.mark.parametrize("mode", ["resize", "letterbox", "augment"])
def test_transform_shape(mode):
    result = make_transform(mode, training=False)(Image.new("RGB", (20, 30)))
    assert result.shape == (3, 224, 224)


def test_dataset_reads_image(tmp_path):
    csv = create_fixture(tmp_path)
    ds = NeoJaundiceDataset(csv, tmp_path, "regression")
    sample = ds[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["target"].item() == pytest.approx(8.5)
    assert sample["patient_id"] == "0001"


def test_classification_target(tmp_path):
    csv = create_fixture(tmp_path)
    assert NeoJaundiceDataset(csv, tmp_path, "classification")[0]["target"].item() == 1


def test_reject_traversal(tmp_path):
    csv = create_fixture(tmp_path)
    frame = pd.read_csv(csv)
    frame.loc[0, "image_path"] = "../outside.jpg"
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        NeoJaundiceDataset(csv, tmp_path, "regression")


def test_model_output():
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 1)


def test_regression_loss():
    loss = calculate_loss(torch.tensor([[2.0]]), torch.tensor([3.0]), "regression")
    assert loss.item() == pytest.approx(0.5)


def test_classification_loss():
    loss = calculate_loss(torch.tensor([[0.0]]), torch.tensor([1.0]), "classification")
    assert loss.item() == pytest.approx(0.693147, abs=1e-5)


def test_patient_metrics():
    rows = [
        {"patient_id": "1", "target": 10.0, "prediction": 9.0},
        {"patient_id": "1", "target": 10.0, "prediction": 11.0},
        {"patient_id": "2", "target": 5.0, "prediction": 6.0},
    ]
    metrics = summarize_predictions(rows, "regression")
    assert metrics["image"]["count"] == 3
    assert metrics["patient"]["count"] == 2
    assert metrics["patient"]["mae"] == pytest.approx(0.5)

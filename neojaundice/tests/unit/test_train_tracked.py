"""Fast unit checks for the tracked trainer; no dataset or GPU required."""

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from scripts.train_tracked import (
    atomic_json_save,
    atomic_torch_save,
    configuration,
    parser,
    validate,
)


def options(tmp_path: Path) -> Namespace:
    return parser().parse_args(
        ["--image-root", str(tmp_path), "--output", str(tmp_path / "run")]
    )


def test_defaults(tmp_path):
    args = options(tmp_path)
    assert args.preprocessing == "augment"
    assert args.task == "regression"
    assert args.patience == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("patience", 0),
        ("batch_size", 0),
        ("workers", -1),
        ("learning_rate", 0),
        ("min_delta", -1),
    ],
)
def test_invalid_arguments(tmp_path, field, value):
    args = options(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate(args)


def test_prevents_overwriting_existing_run(tmp_path):
    args = options(tmp_path)
    args.output.mkdir()
    (args.output / "last_state.pt").touch()
    with pytest.raises(FileExistsError):
        validate(args)


def test_resume_requires_checkpoint(tmp_path):
    args = options(tmp_path)
    args.resume = True
    with pytest.raises(FileNotFoundError):
        validate(args)


def test_resume_config_is_stable(tmp_path):
    args = options(tmp_path)
    assert configuration(args)["seed"] == 42
    assert "epochs" not in configuration(args)


def test_atomic_saves(tmp_path):
    json_path = tmp_path / "history.json"
    torch_path = tmp_path / "checkpoint.pt"
    atomic_json_save({"epoch": 1}, json_path)
    atomic_torch_save({"epoch": 1, "tensor": torch.ones(2)}, torch_path)
    assert '"epoch": 1' in json_path.read_text(encoding="utf-8")
    assert torch.load(torch_path, weights_only=True)["epoch"] == 1
    assert not (tmp_path / "checkpoint.pt.tmp").exists()

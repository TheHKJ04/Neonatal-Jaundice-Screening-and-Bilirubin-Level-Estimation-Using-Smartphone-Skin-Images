"""Tests for the dataset structure analysis CLI."""

from pathlib import Path
from types import SimpleNamespace

from scripts import analyze_dataset_structure


def test_empty_dataset_returns_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "empty_dataset"
    dataset_root.mkdir()

    output_root = tmp_path / "reports"

    monkeypatch.setattr(
        analyze_dataset_structure,
        "parse_arguments",
        lambda: SimpleNamespace(
            input=dataset_root,
            output=output_root,
        ),
    )

    exit_code = analyze_dataset_structure.main()

    assert exit_code == 2
    assert not output_root.exists()

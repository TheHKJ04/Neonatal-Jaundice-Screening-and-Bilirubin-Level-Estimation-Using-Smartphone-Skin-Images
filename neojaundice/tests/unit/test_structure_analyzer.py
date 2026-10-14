"""
Unit tests for NeoJaundice dataset structure analysis.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from ml.datasets.structure_analyzer import (
    analyze_dataset_structure,
    classify_column,
    inspect_csv_header,
    write_structure_json,
    write_structure_markdown,
)


def create_image(path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.new(
        "RGB",
        (32, 32),
        (200, 180, 150),
    ).save(path)


def test_bilirubin_column_detection():
    assert classify_column(
        "Total_Bilirubin",
        ("bilirubin", "tsb", "tcb"),
    )


def test_patient_column_detection():
    assert classify_column(
        "patient_id",
        ("patient", "infant"),
    )


def test_image_column_detection():
    assert classify_column(
        "image_filename",
        ("image", "filename"),
    )


def test_unknown_column_not_matched():
    assert not classify_column(
        "hospital_name",
        ("bilirubin", "tsb", "tcb"),
    )


def test_image_inventory(tmp_path: Path):
    create_image(tmp_path / "a.jpg")
    create_image(tmp_path / "nested" / "b.png")

    report = analyze_dataset_structure(tmp_path)

    assert report.image_files == 2
    assert report.total_files == 2


def test_csv_header_inspection(tmp_path: Path):
    csv_path = tmp_path / "metadata.csv"

    csv_path.write_text(
        "patient_id,image_path,TSB\nP001,image001.jpg,12.5\n",
        encoding="utf-8",
    )

    metadata = inspect_csv_header(
        csv_path,
        tmp_path,
    )

    assert "TSB" in metadata.candidate_bilirubin_columns
    assert "patient_id" in metadata.candidate_patient_columns
    assert "image_path" in metadata.candidate_image_columns


def test_empty_csv_header(tmp_path: Path):
    csv_path = tmp_path / "empty.csv"

    csv_path.write_text(
        "",
        encoding="utf-8",
    )

    metadata = inspect_csv_header(
        csv_path,
        tmp_path,
    )

    assert metadata.inspection_error is not None


def test_excel_file_identified(tmp_path: Path):
    (tmp_path / "metadata.xlsx").write_bytes(b"synthetic placeholder")

    report = analyze_dataset_structure(tmp_path)

    assert report.excel_files == 1
    assert report.metadata_files[0].file_type == "excel"


def test_other_files_counted(tmp_path: Path):
    (tmp_path / "README.txt").write_text(
        "dataset notes",
        encoding="utf-8",
    )

    report = analyze_dataset_structure(tmp_path)

    assert report.other_files == 1


def test_missing_directory(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        analyze_dataset_structure(tmp_path / "missing")


def test_json_report_generation(tmp_path: Path):
    create_image(tmp_path / "sample.png")

    report = analyze_dataset_structure(tmp_path)

    output = tmp_path / "report.json"

    write_structure_json(
        report,
        output,
    )

    content = json.loads(output.read_text(encoding="utf-8"))

    assert content["image_files"] == 1


def test_markdown_report_generation(tmp_path: Path):
    create_image(tmp_path / "sample.png")

    report = analyze_dataset_structure(tmp_path)

    output = tmp_path / "report.md"

    write_structure_markdown(
        report,
        output,
    )

    assert output.exists()

    assert "Dataset Structure Analysis" in output.read_text(encoding="utf-8")


def test_candidate_flags(tmp_path: Path):
    csv_path = tmp_path / "metadata.csv"

    csv_path.write_text(
        "patient_id,image_path,bilirubin_mg_dl\nP001,image001.jpg,12.5\n",
        encoding="utf-8",
    )

    report = analyze_dataset_structure(tmp_path)

    assert report.has_bilirubin_column_candidate
    assert report.has_patient_column_candidate
    assert report.has_image_column_candidate

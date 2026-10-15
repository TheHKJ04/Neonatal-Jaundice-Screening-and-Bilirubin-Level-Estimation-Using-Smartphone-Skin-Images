"""Unit tests for region-specific image EDA."""

from pathlib import Path

import pandas as pd
import pytest

from ml.datasets.region_eda import (
    METRICS,
    build_region_comparison,
    build_region_report,
    calculate_metric_statistics,
    calculate_region_statistics,
    load_image_statistics,
    save_region_reports,
)


def make_dataframe() -> pd.DataFrame:
    """Create a small synthetic image-statistics manifest."""

    rows = []

    for region, brightness_values in (
        ("forehead", [200.0, 220.0]),
        ("sternum", [180.0, 190.0]),
    ):
        for index, brightness in enumerate(
            brightness_values,
            start=1,
        ):
            row = {
                "relative_path": (f"{region}/{index}.png"),
                "body_region_folder": region,
            }

            row.update({metric: 10.0 for metric in METRICS})

            row["brightness"] = brightness

            row["overexposed_fraction"] = 0.0
            row["underexposed_fraction"] = 0.0

            rows.append(row)

    return pd.DataFrame(rows)


def save_dataframe(
    dataframe: pd.DataFrame,
    tmp_path: Path,
) -> Path:
    path = tmp_path / "image_statistics.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


def test_load_valid_statistics(tmp_path: Path):
    path = save_dataframe(
        make_dataframe(),
        tmp_path,
    )

    result = load_image_statistics(path)

    assert len(result) == 4


def test_missing_csv_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_image_statistics(tmp_path / "missing.csv")


def test_missing_column_rejected(tmp_path: Path):
    dataframe = make_dataframe().drop(columns=["brightness"])

    path = save_dataframe(
        dataframe,
        tmp_path,
    )

    with pytest.raises(ValueError):
        load_image_statistics(path)


def test_duplicate_image_paths_rejected(tmp_path: Path):
    dataframe = make_dataframe()

    dataframe.loc[
        1,
        "relative_path",
    ] = dataframe.loc[0, "relative_path"]

    path = save_dataframe(
        dataframe,
        tmp_path,
    )

    with pytest.raises(ValueError):
        load_image_statistics(path)


def test_unknown_region_rejected(tmp_path: Path):
    dataframe = make_dataframe()

    dataframe.loc[
        0,
        "body_region_folder",
    ] = "unknown"

    path = save_dataframe(
        dataframe,
        tmp_path,
    )

    with pytest.raises(ValueError):
        load_image_statistics(path)


def test_missing_numeric_value_rejected(tmp_path: Path):
    dataframe = make_dataframe()

    dataframe.loc[
        0,
        "brightness",
    ] = None

    path = save_dataframe(
        dataframe,
        tmp_path,
    )

    with pytest.raises(ValueError):
        load_image_statistics(path)


def test_non_finite_value_rejected(tmp_path: Path):
    dataframe = make_dataframe()

    dataframe.loc[
        0,
        "brightness",
    ] = float("inf")

    path = save_dataframe(
        dataframe,
        tmp_path,
    )

    with pytest.raises(ValueError):
        load_image_statistics(path)


def test_invalid_exposure_fraction_rejected(tmp_path: Path):
    dataframe = make_dataframe()

    dataframe.loc[
        0,
        "overexposed_fraction",
    ] = 1.5

    path = save_dataframe(
        dataframe,
        tmp_path,
    )

    with pytest.raises(ValueError):
        load_image_statistics(path)


def test_metric_statistics():
    values = pd.Series([10.0, 20.0, 30.0, 40.0])

    result = calculate_metric_statistics(values)

    assert result["mean"] == pytest.approx(25.0)
    assert result["median"] == pytest.approx(25.0)
    assert result["min"] == pytest.approx(10.0)
    assert result["max"] == pytest.approx(40.0)


def test_region_image_counts():
    dataframe = make_dataframe()

    result = calculate_region_statistics(dataframe)

    assert result["forehead"]["image_count"] == 2
    assert result["sternum"]["image_count"] == 2


def test_region_mean_values():
    dataframe = make_dataframe()

    result = calculate_region_statistics(dataframe)

    forehead_mean = result["forehead"]["metrics"]["brightness"]["mean"]

    sternum_mean = result["sternum"]["metrics"]["brightness"]["mean"]

    assert forehead_mean == pytest.approx(210.0)
    assert sternum_mean == pytest.approx(185.0)


def test_mean_difference_direction():
    dataframe = make_dataframe()

    statistics = calculate_region_statistics(dataframe)

    comparison = build_region_comparison(statistics)

    brightness_row = comparison.loc[comparison["metric"] == "brightness"].iloc[0]

    assert brightness_row["mean_difference"] == pytest.approx(25.0)


def test_region_report_structure():
    dataframe = make_dataframe()

    report, comparison = build_region_report(dataframe)

    assert report["total_images"] == 4
    assert "forehead" in report["regions"]
    assert "sternum" in report["regions"]
    assert len(comparison) == len(METRICS)


def test_missing_region_rejected():
    dataframe = make_dataframe()

    dataframe = dataframe.loc[dataframe["body_region_folder"] == "forehead"]

    with pytest.raises(ValueError):
        calculate_region_statistics(dataframe)


def test_report_files_created(tmp_path: Path):
    dataframe = make_dataframe()

    report, comparison = build_region_report(dataframe)

    csv_path, json_path = save_region_reports(
        report,
        comparison,
        tmp_path / "output",
    )

    assert csv_path.is_file()
    assert json_path.is_file()

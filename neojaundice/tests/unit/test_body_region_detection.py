"""Tests for anatomical region detection."""

from pathlib import Path

from ml.datasets.image_eda import detect_body_region


def test_nested_forehead_folder(tmp_path: Path):
    image_path = tmp_path / "source_archive" / "forehead" / "1.jpg"

    result = detect_body_region(
        image_path,
        tmp_path,
    )

    assert result == "forehead"


def test_nested_sternum_folder(tmp_path: Path):
    image_path = tmp_path / "source_archive" / "sternum" / "2.jpg"

    result = detect_body_region(
        image_path,
        tmp_path,
    )

    assert result == "sternum"


def test_unknown_folder(tmp_path: Path):
    image_path = tmp_path / "source_archive" / "images" / "3.jpg"

    result = detect_body_region(
        image_path,
        tmp_path,
    )

    assert result == "unknown"


def test_case_insensitive_detection(tmp_path: Path):
    image_path = tmp_path / "SOURCE_ARCHIVE" / "ForeHead_Images" / "4.jpg"

    result = detect_body_region(
        image_path,
        tmp_path,
    )

    assert result == "forehead"


def test_sternum_name_variation(tmp_path: Path):
    image_path = tmp_path / "source_archive" / "sternum_photos" / "5.jpg"

    result = detect_body_region(
        image_path,
        tmp_path,
    )

    assert result == "sternum"


def test_descriptive_parent_does_not_override_sternum(
    tmp_path: Path,
):
    image_path = (
        tmp_path
        / "source_archive"
        / "Dataset of Forehead and Sternum Images"
        / "Sternum"
        / "1.jpg"
    )

    assert (
        detect_body_region(
            image_path,
            tmp_path,
        )
        == "sternum"
    )


def test_descriptive_parent_does_not_override_forehead(
    tmp_path: Path,
):
    image_path = (
        tmp_path
        / "source_archive"
        / "Dataset of Forehead and Sternum Images"
        / "Forehead"
        / "2.jpg"
    )

    assert (
        detect_body_region(
            image_path,
            tmp_path,
        )
        == "forehead"
    )


def test_descriptive_parent_alone_is_not_a_region(
    tmp_path: Path,
):
    image_path = (
        tmp_path / "source_archive" / "Dataset of Forehead and Sternum Images" / "3.jpg"
    )

    assert (
        detect_body_region(
            image_path,
            tmp_path,
        )
        == "unknown"
    )

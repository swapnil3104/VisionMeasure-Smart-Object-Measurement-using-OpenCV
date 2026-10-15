"""Unit tests for the export_results module (image saving and CSV export)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.export_results import (
    CSV_HEADERS,
    CSVExportError,
    ImageExportError,
    export_csv,
    save_annotated_image,
)
from models.measurement_result import MeasurementResult


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """Return a simple valid 3-channel BGR image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = (255, 255, 255)
    return img


@pytest.fixture
def sample_measurements() -> list[MeasurementResult]:
    """Return a sequence containing calibrated and uncalibrated measurement records."""
    return [
        MeasurementResult(
            object_id=1,
            length_px=200.0,
            width_px=100.0,
            contour_area_px2=20000.0,
            length_cm=4.0,
            width_cm=2.0,
            pixels_per_cm=50.0,
        ),
        MeasurementResult(
            object_id=2,
            length_px=150.0,
            width_px=75.0,
            contour_area_px2=11250.0,
            length_cm=None,
            width_cm=None,
            pixels_per_cm=None,
        ),
    ]


def test_save_annotated_image_png(tmp_path: Path, sample_bgr_image: np.ndarray) -> None:
    """Verify saving a valid annotated image as PNG."""
    out_file = tmp_path / "result.png"
    result_path = save_annotated_image(sample_bgr_image, out_file)
    assert Path(result_path).is_file()
    assert Path(result_path).stat().st_size > 0


def test_save_annotated_image_jpg(tmp_path: Path, sample_bgr_image: np.ndarray) -> None:
    """Verify saving a valid annotated image as JPG."""
    out_file = tmp_path / "result.jpg"
    result_path = save_annotated_image(sample_bgr_image, out_file)
    assert Path(result_path).is_file()
    assert Path(result_path).stat().st_size > 0


def test_save_annotated_image_invalid_extension(tmp_path: Path, sample_bgr_image: np.ndarray) -> None:
    """Verify that unsupported extensions raise ImageExportError."""
    out_file = tmp_path / "result.bmp"
    with pytest.raises(ImageExportError, match="Unsupported image export extension"):
        save_annotated_image(sample_bgr_image, out_file)


def test_save_annotated_image_invalid_image(tmp_path: Path) -> None:
    """Verify that empty or invalid image array raises ImageExportError."""
    out_file = tmp_path / "result.png"
    with pytest.raises(ImageExportError, match="Cannot export invalid image"):
        save_annotated_image(np.array([]), out_file)


def test_save_annotated_image_overwrite_protection(tmp_path: Path, sample_bgr_image: np.ndarray) -> None:
    """Verify that overwrite=False prevents overwriting existing files."""
    out_file = tmp_path / "existing.png"
    out_file.write_text("dummy")
    with pytest.raises(ImageExportError, match="already exists and overwrite is False"):
        save_annotated_image(sample_bgr_image, out_file, overwrite=False)


def test_export_csv_headers_and_values(tmp_path: Path, sample_measurements: list[MeasurementResult]) -> None:
    """Verify that exported CSV contains documented headers and correct record values."""
    csv_file = tmp_path / "test_out.csv"
    res_path = export_csv(
        measurements=sample_measurements,
        output_path=csv_file,
        source_image="test_sample.png",
    )
    assert Path(res_path).is_file()

    with open(res_path, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert reader[0] == CSV_HEADERS
    assert len(reader) == 3  # Header + 2 rows

    # Calibrated row
    row1 = reader[1]
    assert row1[0] == "1"
    assert row1[1] == "200.0"
    assert row1[2] == "100.0"
    assert row1[3] == "20000.0"
    assert row1[4] == "4.0"
    assert row1[5] == "2.0"
    assert row1[6] == "50.0"
    assert row1[7] == "test_sample.png"

    # Uncalibrated row: length_cm, width_cm, pixels_per_cm are empty strings
    row2 = reader[2]
    assert row2[0] == "2"
    assert row2[1] == "150.0"
    assert row2[2] == "75.0"
    assert row2[3] == "11250.0"
    assert row2[4] == ""
    assert row2[5] == ""
    assert row2[6] == ""
    assert row2[7] == "test_sample.png"


def test_export_csv_empty_results(tmp_path: Path) -> None:
    """Verify that exporting empty measurements produces a valid CSV with only headers."""
    csv_file = tmp_path / "empty.csv"
    res_path = export_csv([], csv_file)
    assert Path(res_path).is_file()

    with open(res_path, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert len(reader) == 1
    assert reader[0] == CSV_HEADERS


def test_export_csv_overwrite_protection(tmp_path: Path, sample_measurements: list[MeasurementResult]) -> None:
    """Verify that overwrite=False raises CSVExportError when file already exists."""
    csv_file = tmp_path / "exists.csv"
    csv_file.write_text("existing")
    with pytest.raises(CSVExportError, match="already exists and overwrite is False"):
        export_csv(sample_measurements, csv_file, overwrite=False)


def test_export_csv_invalid_measurement_item(tmp_path: Path) -> None:
    """Verify that non-MeasurementResult items in list raise CSVExportError."""
    csv_file = tmp_path / "invalid.csv"
    with pytest.raises(CSVExportError, match="is not a MeasurementResult instance"):
        export_csv(["not_a_result"], csv_file)  # type: ignore

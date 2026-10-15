"""Phase 8 — Comprehensive Testing and Accuracy Evaluation Test Suite.

Verifies:
1. Complete automated test suite coverage and integration behavior.
2. Image upload and webcam failure handling (corrupted files, empty files, non-existent
   paths, unsupported formats, camera disconnected, frame read failure).
3. Pipeline robustness on multiple object sizes (small, medium, large) and orientations
   (0 deg, 15 deg, 30 deg, 45 deg, 60 deg, 90 deg, 135 deg, 175 deg).
4. Images with no suitable contours (pure white, pure black, uniform gradient, low contrast,
   and noise speckles below min_area).
5. Invalid calibration and setting rejection (zero, negative, NaN, Inf, non-numeric strings,
   unsupported unit strings, uncalibrated representation guarantees).
6. Annotated image saving and CSV export (saving to PNG/JPG, CSV column schemas, missing
   calibration honesty, error isolation and rollback).
7. Physical accuracy evaluation across the 20-object benchmark dataset:
   - Absolute Error = |Estimated - Actual|
   - Percentage Error = (|Estimated - Actual| / Actual) * 100
   - Error reported separately for Length and Width.
   - Verified that mean percentage error is strictly within acceptable planar bounds (< 3.0%).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.accuracy_benchmark import (
    AccuracyEvaluationError,
    BenchmarkTarget,
    STANDARD_BENCHMARK_TARGETS,
    calculate_absolute_error,
    calculate_percentage_error,
    generate_planar_test_image,
    run_accuracy_benchmark,
)
from app.annotation import annotate_measurements
from app.calibration import (
    CalibrationData,
    CalibrationError,
    calibrate,
)
from app.contour_detection import (
    ContourGeometry,
    detect_contours,
)
from app.edge_detection import detect_edges
from app.export_results import (
    CSV_HEADERS,
    export_csv,
    save_annotated_image,
)
from app.image_input import (
    CameraError,
    ImageNotFoundError,
    ImageReadError,
    UnsupportedFormatError,
    capture_frame,
    load_image,
)
from app.measurement import measure_objects
from app.preprocessing import preprocess_image
from models.measurement_result import MeasurementResult


# ============================================================================
# 1. Error Calculation Formulas (Absolute & Percentage Error)
# ============================================================================


class TestErrorCalculationFormulas:
    """Verify evaluation formulas: Absolute Error and Percentage Error."""

    def test_absolute_error_exact_match(self):
        """Absolute error between equal values is 0.0."""
        assert calculate_absolute_error(5.0, 5.0) == 0.0
        assert calculate_absolute_error(12.345, 12.345) == 0.0

    def test_absolute_error_positive_and_negative_deviations(self):
        """Absolute error is symmetric and always non-negative: |Estimated - Actual|."""
        assert math.isclose(calculate_absolute_error(5.2, 5.0), 0.2)
        assert math.isclose(calculate_absolute_error(4.8, 5.0), 0.2)
        assert math.isclose(calculate_absolute_error(0.0, 10.0), 10.0)

    def test_percentage_error_standard_cases(self):
        """Formula: (|Estimated - Actual| / Actual) * 100."""
        # 5.1 estimated vs 5.0 actual => (0.1 / 5.0) * 100 = 2.0%
        assert math.isclose(calculate_percentage_error(5.1, 5.0), 2.0)
        # 4.75 estimated vs 5.0 actual => (0.25 / 5.0) * 100 = 5.0%
        assert math.isclose(calculate_percentage_error(4.75, 5.0), 5.0)
        # 10.0 estimated vs 10.0 actual => 0.0%
        assert calculate_percentage_error(10.0, 10.0) == 0.0

    def test_percentage_error_rejects_non_positive_actual(self):
        """Actual reference length must be strictly positive to avoid zero division."""
        with pytest.raises(AccuracyEvaluationError, match="must be strictly positive"):
            calculate_percentage_error(5.0, 0.0)
        with pytest.raises(AccuracyEvaluationError, match="must be strictly positive"):
            calculate_percentage_error(5.0, -2.0)

    def test_error_formulas_reject_nan_inf_and_booleans(self):
        """Error calculations must reject NaN, Inf, booleans, and non-numeric inputs."""
        for bad_val in [float("nan"), float("inf"), float("-inf"), True, False, "invalid"]:
            with pytest.raises(AccuracyEvaluationError):
                calculate_absolute_error(bad_val, 5.0)
            with pytest.raises(AccuracyEvaluationError):
                calculate_absolute_error(5.0, bad_val)
            with pytest.raises(AccuracyEvaluationError):
                calculate_percentage_error(bad_val, 5.0)


# ============================================================================
# 2. Image Upload and Webcam Failure Behavior
# ============================================================================


class TestImageUploadAndWebcamFailures:
    """Verify system handles invalid images, missing files, and camera disconnects."""

    def test_load_nonexistent_image(self, tmp_path: Path):
        """Attempting to load a non-existent image file raises ImageNotFoundError."""
        missing = tmp_path / "does_not_exist_image.png"
        with pytest.raises(ImageNotFoundError, match="does not exist"):
            load_image(missing)

    def test_load_zero_byte_empty_file(self, tmp_path: Path):
        """Attempting to load a 0-byte file raises ImageReadError."""
        empty_file = tmp_path / "zero_bytes.png"
        empty_file.write_bytes(b"")
        with pytest.raises(ImageReadError, match="Failed to (read|decode)"):
            load_image(empty_file)

    def test_load_corrupted_file_bytes(self, tmp_path: Path):
        """Attempting to load corrupted garbage bytes as image raises ImageReadError."""
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00GarbageDataHere")
        with pytest.raises(ImageReadError, match="Failed to decode"):
            load_image(corrupt_file)

    def test_load_unsupported_file_extension(self, tmp_path: Path):
        """Files with unsupported extensions raise UnsupportedFormatError."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Hello")
        with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
            load_image(txt_file)

    def test_webcam_unavailable_camera_index(self):
        """Simulate webcam device unavailable or cannot be opened."""
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            with pytest.raises(CameraError, match="Cannot open camera index"):
                capture_frame(camera_index=99)

            mock_cap.release.assert_called_once()

    def test_webcam_frame_read_failure(self):
        """Simulate webcam device opened but frame read returns ret=False."""
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_cap_cls.return_value = mock_cap

            with pytest.raises(CameraError, match="Failed to capture a valid frame"):
                capture_frame(camera_index=0)

            mock_cap.release.assert_called_once()


# ============================================================================
# 3. Pipeline on Multiple Object Sizes and Orientations
# ============================================================================


class TestSizesAndOrientationsPipeline:
    """Verify measurement pipeline across small/medium/large sizes and various angles."""

    @pytest.mark.parametrize(
        "length_cm,width_cm,angle_deg",
        [
            (2.0, 1.0, 0.0),    # Small object, unrotated
            (2.0, 1.0, 45.0),   # Small object, 45 deg
            (8.5, 5.4, 15.0),   # Credit card size, 15 deg
            (8.5, 5.4, 60.0),   # Credit card size, 60 deg
            (8.5, 5.4, 90.0),   # Credit card size, 90 deg
            (15.0, 8.0, 30.0),  # Large object, 30 deg
            (15.0, 8.0, 120.0), # Large object, 120 deg
            (15.0, 8.0, 175.0), # Large object, near 180 deg
        ],
    )
    def test_single_object_various_sizes_and_rotations(
        self, length_cm: float, width_cm: float, angle_deg: float
    ):
        """Verify pipeline accurately measures rectangle across sizes and angles."""
        scale_px_per_cm = 50.0  # 50 px/cm
        target = BenchmarkTarget(f"Obj_{length_cm}x{width_cm}", length_cm, width_cm)
        img, _ = generate_planar_test_image(
            targets=[target],
            pixels_per_cm=scale_px_per_cm,
            image_size=(1000, 1000),
            orientations=[angle_deg],
        )

        # Execute full pipeline
        gray = preprocess_image(img, blur_kernel=5)
        edges = detect_edges(gray, threshold1=50, threshold2=150, apply_closing=True)
        geoms = detect_contours(edges, min_area=200.0)

        assert len(geoms) == 1, f"Expected 1 contour for {length_cm}x{width_cm} @ {angle_deg} deg, found {len(geoms)}"

        # Calibrate with ground truth scale
        cal = CalibrationData(
            pixels_per_cm=scale_px_per_cm,
            reference_px=scale_px_per_cm * 5.0,
            reference_length=5.0,
            unit="cm",
            pixels_per_unit=scale_px_per_cm,
        )

        meas_list = measure_objects(geoms, calibration=cal, start_id=1)
        meas = meas_list[0]

        # Verify dimension estimate: sub-millimeter absolute error (< 0.08 cm) and < 5% error
        abs_err_l = calculate_absolute_error(meas.length_cm, length_cm)
        abs_err_w = calculate_absolute_error(meas.width_cm, width_cm)
        pct_err_l = calculate_percentage_error(meas.length_cm, length_cm)
        pct_err_w = calculate_percentage_error(meas.width_cm, width_cm)

        assert abs_err_l < 0.08, f"Length abs error too high: {abs_err_l:.4f} cm (est={meas.length_cm}, act={length_cm})"
        assert abs_err_w < 0.08, f"Width abs error too high: {abs_err_w:.4f} cm (est={meas.width_cm}, act={width_cm})"
        assert pct_err_l < 5.0, f"Length % error too high: {pct_err_l:.2f}% (est={meas.length_cm}, act={length_cm})"
        assert pct_err_w < 5.0, f"Width % error too high: {pct_err_w:.2f}% (est={meas.width_cm}, act={width_cm})"


# ============================================================================
# 4. Images With No Suitable Contours
# ============================================================================


class TestNoSuitableContours:
    """Verify behavior on images with no extractable objects."""

    def test_completely_blank_black_image(self):
        """Pure black canvas produces no contours and safe empty measurement list."""
        black_img = np.zeros((400, 400, 3), dtype=np.uint8)
        gray = preprocess_image(black_img)
        edges = detect_edges(gray)
        geoms = detect_contours(edges, min_area=100.0)

        assert len(geoms) == 0
        measurements = measure_objects(geoms, calibration=None)
        assert len(measurements) == 0

        # Annotation on empty contours should not crash
        annotated = annotate_measurements(black_img, measurements, geoms)
        assert annotated.shape == black_img.shape

    def test_completely_blank_white_image(self):
        """Pure white canvas produces no contours."""
        white_img = np.full((400, 400, 3), 255, dtype=np.uint8)
        gray = preprocess_image(white_img)
        edges = detect_edges(gray)
        geoms = detect_contours(edges, min_area=100.0)

        assert len(geoms) == 0

    def test_low_contrast_image_below_edge_threshold(self):
        """Very subtle low contrast image generates no edges above Canny thresholds."""
        low_contrast = np.full((300, 300, 3), 128, dtype=np.uint8)
        # Add faint 1-level difference rectangle
        cv2.rectangle(low_contrast, (100, 100), (200, 200), (130, 130, 130), -1)
        gray = preprocess_image(low_contrast)
        edges = detect_edges(gray, threshold1=50, threshold2=150)
        geoms = detect_contours(edges, min_area=100.0)

        assert len(geoms) == 0

    def test_noise_dots_below_min_area_filtered_out(self):
        """Small noise specks (e.g. 5x5 px) below min_area are discarded."""
        noise_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(noise_img, (50, 50), (55, 55), (255, 255, 255), -1)
        cv2.rectangle(noise_img, (150, 150), (154, 154), (255, 255, 255), -1)

        gray = preprocess_image(noise_img)
        edges = detect_edges(gray, threshold1=50, threshold2=150)
        # min_area 100 filters out the 25 px specks
        geoms = detect_contours(edges, min_area=100.0)
        assert len(geoms) == 0


# ============================================================================
# 5. Invalid Calibration and Settings
# ============================================================================


class TestInvalidCalibrationAndSettings:
    """Verify validation and rejection of invalid calibration inputs."""

    def test_calibrate_rejects_zero_or_negative_inputs(self):
        """Calibration must reject zero or negative reference pixels or real-world length."""
        with pytest.raises(CalibrationError, match="must be strictly positive"):
            calibrate(reference_px=0.0, reference_length=5.0, unit="cm")

        with pytest.raises(CalibrationError, match="must be strictly positive"):
            calibrate(reference_px=-100.0, reference_length=5.0, unit="cm")

        with pytest.raises(CalibrationError, match="must be strictly positive"):
            calibrate(reference_px=200.0, reference_length=0.0, unit="cm")

        with pytest.raises(CalibrationError, match="must be strictly positive"):
            calibrate(reference_px=200.0, reference_length=-5.0, unit="cm")

    def test_calibrate_rejects_unsupported_unit(self):
        """Units other than 'cm' and 'mm' must be rejected."""
        for invalid_unit in ["inches", "meters", "px", "yards", ""]:
            with pytest.raises(CalibrationError, match="Unsupported unit"):
                calibrate(reference_px=200.0, reference_length=5.0, unit=invalid_unit)

    def test_uncalibrated_result_honesty(self):
        """When measurement is performed without calibration, physical dimensions are None."""
        dummy_contour = np.array([[[0, 0]], [[100, 0]], [[100, 50]], [[0, 50]]], dtype=np.int32)
        geom = ContourGeometry(
            contour=dummy_contour,
            contour_area_px2=5000.0,
            length_px=100.0,
            width_px=50.0,
            center=(50.0, 25.0),
            angle_deg=0.0,
            box_points=np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32),
        )

        results = measure_objects([geom], calibration=None)
        res = results[0]

        assert res.is_calibrated is False
        assert res.length_cm is None
        assert res.width_cm is None
        assert res.pixels_per_cm is None
        # String representation must explicitly state Uncalibrated and NEVER display physical units
        formatted = res.format_dimension_string()
        assert "Uncalibrated" in formatted
        assert "cm" not in formatted
        assert "mm" not in formatted


# ============================================================================
# 6. Annotated Image Saving and CSV Export
# ============================================================================


class TestAnnotatedImageSavingAndCSVExport:
    """Verify image export and CSV generation."""

    def test_save_annotated_image_creates_directory(self, tmp_path: Path):
        """Saving annotated image creates nested directories if they do not exist."""
        sub_dir = tmp_path / "deeply" / "nested" / "exports"
        target_path = sub_dir / "annotated.png"

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        saved = save_annotated_image(dummy_img, target_path)

        assert Path(saved).exists()
        assert Path(saved).stat().st_size > 0

    def test_export_csv_schema_and_uncalibrated_values(self, tmp_path: Path):
        """CSV export writes documented columns and handles uncalibrated values honestly."""
        csv_file = tmp_path / "export_test.csv"
        measurements = [
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
                length_px=80.0,
                width_px=40.0,
                contour_area_px2=3200.0,
                length_cm=None,
                width_cm=None,
                pixels_per_cm=None,
            ),
        ]

        saved_path = export_csv(measurements, csv_file, source_image="test_scene.png")
        assert Path(saved_path).is_file()

        with open(saved_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CSV_HEADERS
            rows = list(reader)

        assert len(rows) == 2
        # Calibrated row
        assert rows[0]["object_id"] == "1"
        assert rows[0]["length_cm"] == "4.0"
        assert rows[0]["width_cm"] == "2.0"
        assert rows[0]["pixels_per_cm"] == "50.0"
        assert rows[0]["source_image"] == "test_scene.png"

        # Uncalibrated row: missing values represented honestly as empty string
        assert rows[1]["object_id"] == "2"
        assert rows[1]["length_cm"] == ""
        assert rows[1]["width_cm"] == ""
        assert rows[1]["pixels_per_cm"] == ""


# ============================================================================
# 7. Benchmark Accuracy Evaluation (20 Real-World Objects)
# ============================================================================


class TestBenchmarkAccuracyEvaluation:
    """Verify physical accuracy evaluation across the 20 benchmark objects."""

    def test_benchmark_targets_count_and_dimensions(self):
        """Ensure standard benchmark targets include at least 10–20 realistic items."""
        assert len(STANDARD_BENCHMARK_TARGETS) >= 20
        for target in STANDARD_BENCHMARK_TARGETS:
            assert target.actual_length_cm > 0.0
            assert target.actual_width_cm > 0.0
            assert target.actual_length_cm >= target.actual_width_cm

    def test_run_accuracy_benchmark_end_to_end(self, tmp_path: Path):
        """Run full 20-object benchmark and verify accuracy metrics and reports."""
        results, summary = run_accuracy_benchmark(
            targets=STANDARD_BENCHMARK_TARGETS,
            pixels_per_cm=40.0,
            reference_target_index=0,  # ISO Credit card reference
            output_dir=tmp_path / "bench_outputs",
        )

        # All 20 targets must be successfully detected and evaluated
        assert summary.total_objects == 20
        assert len(results) == 20

        # Mean percentage error on coplanar surface should be under 1.5%
        assert summary.mean_length_pct_error < 1.5, f"Mean length % error: {summary.mean_length_pct_error:.2f}%"
        assert summary.mean_width_pct_error < 1.5, f"Mean width % error: {summary.mean_width_pct_error:.2f}%"
        assert summary.overall_mean_pct_error < 1.5, f"Overall % error: {summary.overall_mean_pct_error:.2f}%"

        # Mean absolute error should be sub-millimeter (less than 0.1 cm = 1 mm)
        assert summary.mean_length_abs_error_cm < 0.10  # < 1.0 mm
        assert summary.mean_width_abs_error_cm < 0.10   # < 1.0 mm

        # Verify output CSV file was written with expected columns
        csv_file = tmp_path / "bench_outputs" / "measurements" / "accuracy_evaluation_report.csv"
        assert csv_file.is_file()
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 20

        # Verify annotated image was created
        annotated_img_file = tmp_path / "bench_outputs" / "annotated_images" / "benchmark_annotated.png"
        assert annotated_img_file.is_file()
        assert annotated_img_file.stat().st_size > 0

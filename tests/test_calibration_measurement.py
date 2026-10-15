"""Tests for Phase 4: Calibration and Measurement Engine.

Covers:
- Pixel-per-unit scale calculation and unit normalization (cm and mm).
- Required verification ratios:
    - 250 reference pixels / 5 cm = 50 pixels/cm.
    - 400 pixels / 50 pixels/cm = 8 cm.
    - 70 pixels / 50 pixels/cm = 1.4 cm.
- Rejection of zero, negative, NaN, Inf, and non-numeric inputs.
- Consistent mm and cm conversions.
- Uncalibrated behavior (no silent fabricated defaults).
- Object measurement from ContourGeometry and degenerate geometry handling.
- Sequential measuring and empty geometry handling.
- Planar constraint documentation and guarantees.
"""
import numpy as np
import pytest

from app.calibration import (
    CalibrationError,
    calculate_pixel_distance,
    calculate_pixels_per_unit,
    calibrate,
    calibrate_from_geometry,
    calibrate_from_points,
    normalize_unit,
)
from app.contour_detection import ContourGeometry
from app.measurement import (
    MeasurementError,
    calculate_physical_dimension,
    measure_object,
    measure_objects,
)
from models.measurement_result import MeasurementResult


def create_mock_geometry(
    length_px: float = 140.0,
    width_px: float = 70.0,
    area_px2: float = 9800.0,
) -> ContourGeometry:
    """Helper to construct a valid ContourGeometry instance for testing."""
    dummy_contour = np.array([[[0, 0]], [[140, 0]], [[140, 70]], [[0, 70]]], dtype=np.int32)
    box_pts = np.array([[0, 0], [140, 0], [140, 70], [0, 70]], dtype=np.float32)
    return ContourGeometry(
        contour=dummy_contour,
        contour_area_px2=area_px2,
        length_px=length_px,
        width_px=width_px,
        center=(70.0, 35.0),
        angle_deg=0.0,
        box_points=box_pts,
    )


# ============================================================================
# Calibration Engine Tests
# ============================================================================


def test_calculate_pixels_per_unit_required_standard():
    """Verify standard formula: 250 reference pixels / 5 cm = 50 pixels/cm."""
    scale = calculate_pixels_per_unit(reference_px=250.0, reference_length=5.0, unit="cm")
    assert pytest.approx(scale, rel=1e-5) == 50.0


def test_calibrate_cm_and_mm_consistency():
    """Verify that 5 cm and 50 mm produce consistent scale factors.

    250 px / 5 cm -> 50 px/cm, 5 px/mm.
    250 px / 50 mm -> 5 px/mm, 50 px/cm.
    """
    cal_cm = calibrate(reference_px=250.0, reference_length=5.0, unit="cm")
    assert pytest.approx(cal_cm.pixels_per_cm, rel=1e-5) == 50.0
    assert pytest.approx(cal_cm.pixels_per_mm, rel=1e-5) == 5.0
    assert pytest.approx(cal_cm.pixels_per_unit, rel=1e-5) == 50.0
    assert cal_cm.unit == "cm"

    cal_mm = calibrate(reference_px=250.0, reference_length=50.0, unit="mm")
    assert pytest.approx(cal_mm.pixels_per_mm, rel=1e-5) == 5.0
    assert pytest.approx(cal_mm.pixels_per_cm, rel=1e-5) == 50.0
    assert pytest.approx(cal_mm.pixels_per_unit, rel=1e-5) == 5.0
    assert cal_mm.unit == "mm"

    # Conversion methods on CalibrationData
    assert pytest.approx(cal_cm.pixels_to_cm(400.0), rel=1e-5) == 8.0
    assert pytest.approx(cal_cm.pixels_to_mm(400.0), rel=1e-5) == 80.0
    assert pytest.approx(cal_mm.pixels_to_cm(70.0), rel=1e-5) == 1.4
    assert pytest.approx(cal_mm.pixels_to_mm(70.0), rel=1e-5) == 14.0


def test_unit_normalization():
    """Verify case insensitivity and whitespace stripping for units."""
    assert normalize_unit(" cm ") == "cm"
    assert normalize_unit("CM") == "cm"
    assert normalize_unit("Mm") == "mm"
    assert normalize_unit("  MM  ") == "mm"

    with pytest.raises(CalibrationError, match="Unsupported unit"):
        normalize_unit("inches")

    with pytest.raises(CalibrationError, match="Unsupported unit"):
        normalize_unit("m")

    with pytest.raises(CalibrationError, match="Unit must be a string"):
        normalize_unit(123)  # type: ignore


@pytest.mark.parametrize("invalid_ref_px", [0, -1, -250.0])
def test_reject_zero_or_negative_reference_pixels(invalid_ref_px):
    """Zero or negative pixel span must be rejected."""
    with pytest.raises(CalibrationError, match="Reference pixel span must be strictly positive"):
        calibrate(reference_px=invalid_ref_px, reference_length=5.0, unit="cm")


@pytest.mark.parametrize("invalid_ref_len", [0, -1, -5.0])
def test_reject_zero_or_negative_reference_length(invalid_ref_len):
    """Zero or negative reference length must be rejected."""
    with pytest.raises(CalibrationError, match="Reference length must be strictly positive"):
        calibrate(reference_px=250.0, reference_length=invalid_ref_len, unit="cm")


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf"), True, False, "50"])
def test_reject_non_finite_or_non_numeric_calibration_inputs(bad_val):
    """Reject NaN, inf, bools, and string numeric types."""
    with pytest.raises(CalibrationError):
        calibrate(reference_px=bad_val, reference_length=5.0)

    with pytest.raises(CalibrationError):
        calibrate(reference_px=250.0, reference_length=bad_val)


def test_calibrate_from_geometry():
    """Verify calibration directly from a detected ContourGeometry."""
    geom = create_mock_geometry(length_px=250.0, width_px=100.0)

    # Calibrate using longer side
    cal_longer = calibrate_from_geometry(geom, reference_length=5.0, unit="cm", side="longer")
    assert pytest.approx(cal_longer.pixels_per_cm, rel=1e-5) == 50.0

    # Calibrate using shorter side
    cal_shorter = calibrate_from_geometry(geom, reference_length=2.0, unit="cm", side="shorter")
    assert pytest.approx(cal_shorter.pixels_per_cm, rel=1e-5) == 50.0

    # Reject invalid side
    with pytest.raises(CalibrationError, match="Invalid reference side"):
        calibrate_from_geometry(geom, reference_length=5.0, side="diagonal")

    # Reject None geometry
    with pytest.raises(CalibrationError, match="geometry cannot be None"):
        calibrate_from_geometry(None, reference_length=5.0)  # type: ignore


def test_calculate_pixel_distance_and_points_calibration():
    """Verify euclidean point distance and calibration from two clicked points."""
    p1 = (100.0, 100.0)
    p2 = (350.0, 100.0)  # 250 px horizontal distance
    dist = calculate_pixel_distance(p1, p2)
    assert pytest.approx(dist, rel=1e-5) == 250.0

    cal = calibrate_from_points(p1, p2, reference_length=5.0, unit="cm")
    assert pytest.approx(cal.pixels_per_cm, rel=1e-5) == 50.0

    # Reject identical points (zero distance)
    with pytest.raises(CalibrationError, match="Distance between calibration points cannot be zero"):
        calculate_pixel_distance((100, 100), (100, 100))

    # Reject invalid point structures
    with pytest.raises(CalibrationError):
        calculate_pixel_distance((100,), (200, 200))  # type: ignore


# ============================================================================
# Measurement Engine Tests
# ============================================================================


def test_calculate_physical_dimension_required_standards():
    """Verify the minimum required test formulas:

    400 pixels / 50 pixels/cm = 8 cm.
    70 pixels / 50 pixels/cm = 1.4 cm.
    """
    length_cm = calculate_physical_dimension(pixel_length=400.0, pixels_per_unit=50.0)
    assert pytest.approx(length_cm, rel=1e-5) == 8.0

    width_cm = calculate_physical_dimension(pixel_length=70.0, pixels_per_unit=50.0)
    assert pytest.approx(width_cm, rel=1e-5) == 1.4


def test_calculate_physical_dimension_validation():
    """Reject negative lengths or non-positive scales."""
    with pytest.raises(MeasurementError, match="Pixel length cannot be negative"):
        calculate_physical_dimension(-10.0, 50.0)

    with pytest.raises(MeasurementError, match="Scale factor.*must be strictly positive"):
        calculate_physical_dimension(100.0, 0.0)

    with pytest.raises(MeasurementError, match="Scale factor.*must be strictly positive"):
        calculate_physical_dimension(100.0, -50.0)

    with pytest.raises(MeasurementError):
        calculate_physical_dimension(float("nan"), 50.0)


def test_measure_object_calibrated():
    """Verify measurement with valid calibration produces correct cm and mm values."""
    geom = create_mock_geometry(length_px=400.0, width_px=70.0, area_px2=28000.0)
    cal = calibrate(reference_px=250.0, reference_length=5.0, unit="cm")  # 50 px/cm

    res = measure_object(geometry=geom, object_id=1, calibration=cal)

    assert isinstance(res, MeasurementResult)
    assert res.object_id == 1
    assert res.length_px == 400.0
    assert res.width_px == 70.0
    assert res.contour_area_px2 == 28000.0

    # Calibrated values
    assert pytest.approx(res.length_cm, rel=1e-5) == 8.0
    assert pytest.approx(res.width_cm, rel=1e-5) == 1.4
    assert pytest.approx(res.length_mm, rel=1e-5) == 80.0
    assert pytest.approx(res.width_mm, rel=1e-5) == 14.0
    assert pytest.approx(res.pixels_per_cm, rel=1e-5) == 50.0
    assert pytest.approx(res.pixels_per_mm, rel=1e-5) == 5.0
    assert res.is_calibrated is True


def test_measure_object_uncalibrated_explicit_none():
    """Verify uncalibrated state explicitly sets physical dimensions to None without fabricating numbers."""
    geom = create_mock_geometry(length_px=400.0, width_px=70.0, area_px2=28000.0)

    res = measure_object(geometry=geom, object_id=1, calibration=None)

    assert res.object_id == 1
    assert res.length_px == 400.0
    assert res.width_px == 70.0
    assert res.contour_area_px2 == 28000.0

    # Physical dimensions must be explicitly None
    assert res.length_cm is None
    assert res.width_cm is None
    assert res.length_mm is None
    assert res.width_mm is None
    assert res.pixels_per_cm is None
    assert res.pixels_per_mm is None
    assert res.is_calibrated is False


def test_measure_object_rejects_invalid_or_degenerate_geometry():
    """Reject degenerate geometry such as zero or negative side lengths."""
    cal = calibrate(reference_px=50.0, reference_length=1.0)

    # None geometry
    with pytest.raises(MeasurementError, match="Contour geometry cannot be None"):
        measure_object(geometry=None, object_id=1, calibration=cal)  # type: ignore

    # Zero length_px
    zero_len_geom = create_mock_geometry(length_px=0.0, width_px=50.0)
    with pytest.raises(MeasurementError, match="Degenerate geometry"):
        measure_object(geometry=zero_len_geom, object_id=1, calibration=cal)

    # Negative width_px
    neg_width_geom = create_mock_geometry(length_px=100.0, width_px=-10.0)
    with pytest.raises(MeasurementError, match="Degenerate geometry"):
        measure_object(geometry=neg_width_geom, object_id=1, calibration=cal)

    # Negative contour area
    neg_area_geom = create_mock_geometry(length_px=100.0, width_px=50.0, area_px2=-5.0)
    with pytest.raises(MeasurementError, match="contour area cannot be negative"):
        measure_object(geometry=neg_area_geom, object_id=1, calibration=cal)

    # Invalid object ID
    valid_geom = create_mock_geometry()
    with pytest.raises(MeasurementError, match="Object ID must be a non-negative integer"):
        measure_object(geometry=valid_geom, object_id=-1, calibration=cal)

    # Invalid calibration type
    with pytest.raises(MeasurementError, match="Expected calibration to be CalibrationData"):
        measure_object(geometry=valid_geom, object_id=1, calibration="invalid")  # type: ignore


def test_measure_objects_batch_and_empty():
    """Verify batch measurement with sequential IDs and empty list handling."""
    cal = calibrate(reference_px=250.0, reference_length=5.0)  # 50 px/cm

    geometries = [
        create_mock_geometry(length_px=400.0, width_px=200.0),
        create_mock_geometry(length_px=150.0, width_px=75.0),
    ]

    results = measure_objects(geometries, calibration=cal, start_id=1)
    assert len(results) == 2

    assert results[0].object_id == 1
    assert pytest.approx(results[0].length_cm, rel=1e-5) == 8.0
    assert pytest.approx(results[0].width_cm, rel=1e-5) == 4.0

    assert results[1].object_id == 2
    assert pytest.approx(results[1].length_cm, rel=1e-5) == 3.0
    assert pytest.approx(results[1].width_cm, rel=1e-5) == 1.5

    # Empty geometry list returns empty results list without error
    assert measure_objects([], calibration=cal) == []

    # Uncalibrated batch measurement
    uncal_results = measure_objects(geometries, calibration=None)
    assert len(uncal_results) == 2
    assert uncal_results[0].is_calibrated is False
    assert uncal_results[1].is_calibrated is False


def test_planar_constraint_documentation_present():
    """Verify that module documentation clearly explains the planar constraint requirement."""
    import app.calibration as cal_mod
    import app.measurement as meas_mod

    assert "planar" in cal_mod.__doc__.lower()
    assert "working plane" in cal_mod.__doc__.lower()
    assert "perspective" in cal_mod.__doc__.lower()

    assert "planar" in meas_mod.__doc__.lower()
    assert "plane" in meas_mod.__doc__.lower()

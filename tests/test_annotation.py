"""Unit tests for Phase 5: Annotation and Results Model.

Tests cover:
- Preserving source image unmodified (non-destructive guarantee).
- Correct dimensional values and unit presentation (cm and mm).
- Uncalibrated results strictly avoiding physical units and displaying px + (Uncalibrated).
- Rotated bounding box rendering and midpoint indicators.
- Empty result lists producing a valid image and an explicit empty state.
- Label placement avoiding object boundaries and clamping within canvas bounds.
- Structured MeasurementResult model dictionary export and formatting.
- Error handling on invalid images, degenerate inputs, and missing box coordinates.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.annotation import (
    AnnotationConfig,
    AnnotationError,
    annotate_measurements,
    calculate_label_position,
    draw_edge_midpoints,
    draw_rotated_box,
    format_annotation_text,
)
from app.contour_detection import ContourGeometry
from models.measurement_result import MeasurementResult


def create_sample_box_points(
    center_x: float = 200.0,
    center_y: float = 150.0,
    width: float = 60.0,
    height: float = 120.0,
) -> np.ndarray:
    """Helper to generate a 4-point rectangle."""
    half_w = width / 2.0
    half_h = height / 2.0
    return np.array(
        [
            [center_x - half_w, center_y - half_h],
            [center_x + half_w, center_y - half_h],
            [center_x + half_w, center_y + half_h],
            [center_x - half_w, center_y + half_h],
        ],
        dtype=np.float32,
    )


def create_sample_measurement(
    object_id: int = 1,
    length_px: float = 400.0,
    width_px: float = 70.0,
    area_px2: float = 28000.0,
    length_cm: float | None = 8.0,
    width_cm: float | None = 1.4,
    pixels_per_cm: float | None = 50.0,
    box_points: np.ndarray | None = None,
) -> MeasurementResult:
    """Helper to generate a test MeasurementResult."""
    if box_points is None:
        box_points = create_sample_box_points()

    return MeasurementResult(
        object_id=object_id,
        length_px=length_px,
        width_px=width_px,
        contour_area_px2=area_px2,
        length_cm=length_cm,
        width_cm=width_cm,
        pixels_per_cm=pixels_per_cm,
        box_points=box_points,
        center=(float(np.mean(box_points[:, 0])), float(np.mean(box_points[:, 1]))),
    )


# ============================================================================
# Non-destructive Image Tests
# ============================================================================


def test_annotation_leaves_color_source_image_unmodified():
    """Verify that annotate_measurements does not modify a 3-channel input array."""
    # Create black canvas
    original = np.zeros((400, 500, 3), dtype=np.uint8)
    original_copy = original.copy()

    meas = create_sample_measurement(object_id=1)
    annotated = annotate_measurements(original, [meas])

    # Original array must be completely untouched
    assert np.array_equal(original, original_copy)
    assert original is not annotated
    # Annotated image must have drawings (not all zeros)
    assert np.any(annotated > 0)


def test_annotation_leaves_grayscale_source_image_unmodified():
    """Verify that annotate_measurements handles 2D grayscale images non-destructively."""
    original_gray = np.zeros((400, 500), dtype=np.uint8)
    original_copy = original_gray.copy()

    meas = create_sample_measurement(object_id=1)
    annotated = annotate_measurements(original_gray, [meas])

    assert np.array_equal(original_gray, original_copy)
    assert original_gray.ndim == 2
    assert annotated.ndim == 3  # Returned as BGR for color rendering
    assert annotated.shape[:2] == (400, 500)
    assert np.any(annotated > 0)


# ============================================================================
# Units and Text Formatting Tests
# ============================================================================


def test_format_annotation_text_calibrated_cm():
    """Verify calibrated results display centimeters accurately."""
    meas = create_sample_measurement(
        object_id=1,
        length_px=400.0,
        width_px=70.0,
        length_cm=8.0,
        width_cm=1.4,
        pixels_per_cm=50.0,
    )
    text = format_annotation_text(meas, preferred_unit="cm")
    assert "Obj 1:" in text
    assert "8.00 x 1.40 cm" in text
    assert "px" not in text
    assert "Uncalibrated" not in text


def test_format_annotation_text_calibrated_mm():
    """Verify calibrated results convert and display millimeters accurately."""
    meas = create_sample_measurement(
        object_id=2,
        length_px=400.0,
        width_px=70.0,
        length_cm=8.0,
        width_cm=1.4,
        pixels_per_cm=50.0,
    )
    text = format_annotation_text(meas, preferred_unit="mm")
    assert "Obj 2:" in text
    assert "80.0 x 14.0 mm" in text
    assert "cm" not in text
    assert "px" not in text


def test_format_annotation_text_uncalibrated_never_shows_physical_units():
    """Verify uncalibrated results NEVER display physical units even if preferred_unit is passed."""
    uncal = create_sample_measurement(
        object_id=3,
        length_px=400.0,
        width_px=70.0,
        length_cm=None,
        width_cm=None,
        pixels_per_cm=None,
    )

    # Test with default cm
    text_cm = format_annotation_text(uncal, preferred_unit="cm")
    assert "Obj 3:" in text_cm
    assert "400.0 x 70.0 px" in text_cm
    assert "(Uncalibrated)" in text_cm
    assert "cm" not in text_cm
    assert "mm" not in text_cm

    # Test with explicit mm
    text_mm = format_annotation_text(uncal, preferred_unit="mm")
    assert "400.0 x 70.0 px" in text_mm
    assert "(Uncalibrated)" in text_mm
    assert "mm" not in text_mm
    assert "cm" not in text_mm


# ============================================================================
# Empty Results Handling
# ============================================================================


def test_empty_measurements_returns_valid_image_with_clear_no_results_state():
    """Verify empty measurement list produces a valid image and a clear no-results state."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # With empty message banner enabled (default)
    annotated = annotate_measurements(img, [])
    assert annotated is not img
    assert annotated.shape == (300, 400, 3)
    assert annotated.dtype == np.uint8
    # Banner was rendered
    assert np.any(annotated > 0)

    # With empty message disabled
    cfg = AnnotationConfig(show_empty_message=False)
    annotated_no_banner = annotate_measurements(img, [], config=cfg)
    assert np.array_equal(annotated_no_banner, img)


# ============================================================================
# Label Positioning and Boundary Avoidance
# ============================================================================


def test_label_position_placed_above_box_normally():
    """Verify badge is positioned above the topmost boundary when headroom exists."""
    # Box located around y = 200..300 in a 600h image
    box_pts = create_sample_box_points(center_x=300, center_y=250, width=100, height=80)
    text_size = (120, 20)
    image_shape = (600, 600, 3)

    text_x, text_y = calculate_label_position(
        box_points=box_pts,
        text_size=text_size,
        image_shape=image_shape,
        offset_y=8,
    )

    min_y = float(np.min(box_pts[:, 1]))  # 250 - 40 = 210
    # The top of the badge must be strictly above min_y
    badge_top = text_y - text_size[1] - 4
    assert badge_top < min_y


def test_label_position_repositions_when_near_top_boundary():
    """Verify badge repositions below or inside the box when near the top image border."""
    # Box located near top edge: y = 2..50
    box_pts = np.array([[50, 2], [150, 2], [150, 50], [50, 50]], dtype=np.float32)
    text_size = (100, 15)
    image_shape = (400, 400, 3)

    text_x, text_y = calculate_label_position(
        box_points=box_pts,
        text_size=text_size,
        image_shape=image_shape,
        offset_y=8,
    )

    # Must not have negative coordinates
    badge_top = text_y - text_size[1] - 4
    assert badge_top >= 0
    # Placed below the box since min_y = 2 has no room above
    max_y = float(np.max(box_pts[:, 1]))
    assert text_y > max_y


def test_label_position_clamped_horizontally():
    """Verify label origin is clamped inside image canvas horizontally."""
    # Box on the far right edge
    box_pts = np.array([[380, 100], [398, 100], [398, 200], [380, 200]], dtype=np.float32)
    text_size = (150, 20)
    image_shape = (300, 400, 3)

    text_x, text_y = calculate_label_position(
        box_points=box_pts,
        text_size=text_size,
        image_shape=image_shape,
    )

    # Text must fit inside 400px width
    assert text_x >= 4
    assert text_x + text_size[0] <= 400


# ============================================================================
# Drawing Components Tests
# ============================================================================


def test_draw_rotated_box():
    """Verify rotated box is drawn onto the canvas."""
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    box_pts = create_sample_box_points(center_x=100, center_y=100, width=50, height=80)

    draw_rotated_box(canvas, box_pts, color=(0, 255, 0), thickness=2)
    assert np.any(canvas[:, :, 1] == 255)  # Green channel active


def test_draw_edge_midpoints():
    """Verify edge midpoint markers are drawn."""
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    box_pts = create_sample_box_points(center_x=100, center_y=100, width=60, height=60)

    draw_edge_midpoints(canvas, box_pts, color=(0, 165, 255), radius=3)
    # Check that midpoint color is present
    assert np.any(canvas[:, :, 0] == 0)
    assert np.any(canvas[:, :, 1] == 165)
    assert np.any(canvas[:, :, 2] == 255)


# ============================================================================
# Batch and Integration with Pipeline
# ============================================================================


def test_batch_annotation_pipeline():
    """Verify batch annotation with multiple objects and mixed calibration."""
    img = np.zeros((500, 600, 3), dtype=np.uint8)

    pts1 = create_sample_box_points(center_x=150, center_y=150, width=80, height=140)
    pts2 = create_sample_box_points(center_x=400, center_y=350, width=100, height=60)

    # Object 1: calibrated
    m1 = create_sample_measurement(
        object_id=1,
        length_px=140.0,
        width_px=80.0,
        length_cm=2.8,
        width_cm=1.6,
        pixels_per_cm=50.0,
        box_points=pts1,
    )
    # Object 2: uncalibrated
    m2 = create_sample_measurement(
        object_id=2,
        length_px=100.0,
        width_px=60.0,
        length_cm=None,
        width_cm=None,
        pixels_per_cm=None,
        box_points=pts2,
    )

    annotated = annotate_measurements(img, [m1, m2], preferred_unit="cm")
    assert annotated.shape == img.shape
    assert np.any(annotated > 0)


def test_annotation_with_separate_geometries_list():
    """Verify annotation works when box_points are supplied via geometries parameter."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    dummy_cnt = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)
    box_pts = create_sample_box_points(center_x=150, center_y=150, width=100, height=100)

    geom = ContourGeometry(
        contour=dummy_cnt,
        contour_area_px2=10000.0,
        length_px=100.0,
        width_px=100.0,
        center=(150.0, 150.0),
        angle_deg=0.0,
        box_points=box_pts,
    )

    # Measurement without box_points
    meas = MeasurementResult(
        object_id=1,
        length_px=100.0,
        width_px=100.0,
        contour_area_px2=10000.0,
        length_cm=2.0,
        width_cm=2.0,
        pixels_per_cm=50.0,
        box_points=None,
    )

    annotated = annotate_measurements(img, [meas], geometries=[geom])
    assert np.any(annotated > 0)


# ============================================================================
# MeasurementResult Model Tests
# ============================================================================


def test_measurement_result_to_dict():
    """Verify MeasurementResult to_dict returns properly rounded fields."""
    meas = create_sample_measurement(
        object_id=1,
        length_px=400.123,
        width_px=70.456,
        area_px2=28000.789,
        length_cm=8.0024,
        width_cm=1.4091,
        pixels_per_cm=50.0,
    )
    d = meas.to_dict()
    assert d["object_id"] == 1
    assert d["length_px"] == 400.12
    assert d["width_px"] == 70.46
    assert d["contour_area_px2"] == 28000.79
    assert d["is_calibrated"] is True
    assert d["length_cm"] == 8.002
    assert d["width_cm"] == 1.409
    assert d["length_mm"] == 80.02
    assert d["width_mm"] == 14.09
    assert d["pixels_per_cm"] == 50.0


def test_measurement_result_format_dimension_string_options():
    """Verify format_dimension_string options (include_id, unit)."""
    meas = create_sample_measurement(
        object_id=5,
        length_px=400.0,
        width_px=70.0,
        length_cm=8.0,
        width_cm=1.4,
        pixels_per_cm=50.0,
    )
    assert meas.format_dimension_string(include_id=True, unit="cm") == "Obj 5: 8.00 x 1.40 cm"
    assert meas.format_dimension_string(include_id=False, unit="cm") == "8.00 x 1.40 cm"
    assert meas.format_dimension_string(include_id=True, unit="mm") == "Obj 5: 80.0 x 14.0 mm"
    assert meas.format_dimension_string(include_id=False, unit="mm") == "80.0 x 14.0 mm"


# ============================================================================
# Error Handling and Rejection Tests
# ============================================================================


def test_annotation_rejects_none_or_empty_image():
    """Verify AnnotationError on invalid image inputs."""
    with pytest.raises(AnnotationError, match="must be a valid NumPy ndarray"):
        annotate_measurements(None, [])  # type: ignore

    with pytest.raises(AnnotationError, match="empty"):
        annotate_measurements(np.array([], dtype=np.uint8), [])

    with pytest.raises(AnnotationError, match="dtype uint8"):
        annotate_measurements(np.zeros((100, 100), dtype=np.float32), [])

    with pytest.raises(AnnotationError, match="dimensions"):
        annotate_measurements(np.zeros((10, 10, 10, 10), dtype=np.uint8), [])


def test_annotation_rejects_non_measurement_item():
    """Verify AnnotationError when measurements list contains unexpected types."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(AnnotationError, match="MeasurementResult"):
        annotate_measurements(img, ["not_a_result"])  # type: ignore


def test_annotation_rejects_missing_box_points():
    """Verify AnnotationError when neither measurement nor geometry has box points."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    meas = MeasurementResult(
        object_id=1,
        length_px=100.0,
        width_px=50.0,
        contour_area_px2=5000.0,
        box_points=None,
    )
    with pytest.raises(AnnotationError, match="does not have bounding box points"):
        annotate_measurements(img, [meas])

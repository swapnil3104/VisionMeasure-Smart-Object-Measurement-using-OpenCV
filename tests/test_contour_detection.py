"""Tests for contour detection and geometry module (Phase 3)."""
import cv2
import numpy as np
import pytest

from app.contour_detection import (
    ContourDetectionError,
    calculate_contour_geometry,
    detect_contours,
    extract_external_contours,
    filter_contours_by_area,
)
from models.measurement_result import MeasurementResult


def create_solid_rectangle_edge_image(
    image_shape: tuple[int, int] = (400, 400),
    rect_center: tuple[float, float] = (200.0, 200.0),
    rect_size: tuple[float, float] = (120.0, 60.0),
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Generate a clean binary edge image of a rotated solid rectangle.

    Args:
        image_shape: (height, width) of the generated image.
        rect_center: (cx, cy) center of the rectangle.
        rect_size: (width, height) unrotated rectangle dimensions.
        angle_deg: Rotation angle in degrees.

    Returns:
        2D single-channel binary edge image (uint8) where edges are 255.
    """
    canvas = np.zeros(image_shape, dtype=np.uint8)
    rect = (rect_center, rect_size, angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(canvas, [box], 0, 255, thickness=-1)
    # Extract clean edges via Canny and close small corner gaps
    edges = cv2.Canny(canvas, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


# ============================================================================
# Basic Detection & Normalization Tests
# ============================================================================


def test_detect_contours_basic_rectangle():
    """Verify a clear rectangular target produces a valid candidate contour with normalized sides."""
    # Target 140 px by 70 px unrotated
    edge_image = create_solid_rectangle_edge_image(
        image_shape=(300, 300),
        rect_center=(150.0, 150.0),
        rect_size=(140.0, 70.0),
        angle_deg=0.0,
    )

    geometries = detect_contours(edge_image, min_area=500.0)

    assert len(geometries) == 1
    geom = geometries[0]

    # Verify normalization: length_px must always be >= width_px
    assert geom.length_px >= geom.width_px
    # Allow small tolerance (+- 3px) for edge rasterization
    assert abs(geom.length_px - 140.0) < 3.0
    assert abs(geom.width_px - 70.0) < 3.0

    # Area verification (~ 140 * 70 = 9800)
    assert abs(geom.contour_area_px2 - 9800.0) < 400.0

    # Bounding box points structure
    assert geom.box_points.shape == (4, 2)
    assert geom.box_points.dtype == np.float32

    # Centroid
    cx, cy = geom.center
    assert abs(cx - 150.0) < 2.0
    assert abs(cy - 150.0) < 2.0


@pytest.mark.parametrize("angle", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 135.0])
def test_detect_contours_rotated_rectangles(angle: float):
    """Verify rotated objects are handled accurately and side lengths remain normalized."""
    target_long = 120.0
    target_short = 50.0

    edge_image = create_solid_rectangle_edge_image(
        image_shape=(400, 400),
        rect_center=(200.0, 200.0),
        rect_size=(target_long, target_short),
        angle_deg=angle,
    )

    geometries = detect_contours(edge_image, min_area=500.0)

    assert len(geometries) == 1
    geom = geometries[0]

    # Ensure side lengths are normalized consistently regardless of rotation
    assert geom.length_px >= geom.width_px
    assert abs(geom.length_px - target_long) < 4.0
    assert abs(geom.width_px - target_short) < 4.0


@pytest.mark.parametrize(
    "long_side,short_side",
    [
        (220.0, 110.0),  # Large
        (80.0, 40.0),    # Medium
        (40.0, 20.0),    # Small
    ],
)
def test_detect_contours_different_sizes(long_side: float, short_side: float):
    """Verify detection accuracy across multiple rectangle scales."""
    edge_image = create_solid_rectangle_edge_image(
        image_shape=(400, 400),
        rect_center=(200.0, 200.0),
        rect_size=(long_side, short_side),
        angle_deg=25.0,
    )

    geometries = detect_contours(edge_image, min_area=300.0)

    assert len(geometries) == 1
    geom = geometries[0]

    assert geom.length_px >= geom.width_px
    assert abs(geom.length_px - long_side) < 4.0
    assert abs(geom.width_px - short_side) < 4.0


# ============================================================================
# Filtering and Edge Case Tests
# ============================================================================


def test_minimum_area_filtering():
    """Verify small contours can be excluded using configurable min_area."""
    canvas = np.zeros((400, 500), dtype=np.uint8)

    # Large rectangle: 120 x 80 = 9600 area
    box_large = cv2.boxPoints(((150.0, 200.0), (120.0, 80.0), 0.0)).astype(np.int32)
    cv2.drawContours(canvas, [box_large], 0, 255, -1)

    # Small rectangle: 30 x 20 = 600 area
    box_small = cv2.boxPoints(((380.0, 200.0), (30.0, 20.0), 0.0)).astype(np.int32)
    cv2.drawContours(canvas, [box_small], 0, 255, -1)

    edge_image = cv2.Canny(canvas, 50, 150)

    # Low threshold: both candidates detected
    geoms_all = detect_contours(edge_image, min_area=200.0)
    assert len(geoms_all) == 2

    # Medium threshold: only large candidate detected
    geoms_large = detect_contours(edge_image, min_area=2000.0)
    assert len(geoms_large) == 1
    assert abs(geoms_large[0].length_px - 120.0) < 4.0

    # Very high threshold: no candidates accepted
    geoms_none = detect_contours(edge_image, min_area=20000.0)
    assert len(geoms_none) == 0


def test_empty_edge_image_returns_empty_list():
    """Verify an edge image with no edges/contours returns an empty list without crashing."""
    empty_edges = np.zeros((300, 300), dtype=np.uint8)
    results = detect_contours(empty_edges, min_area=100.0)
    assert results == []


def test_extract_external_contours_does_not_mutate_input():
    """Verify the original edge image is not modified during contour extraction."""
    edge_image = create_solid_rectangle_edge_image(
        image_shape=(200, 200),
        rect_center=(100.0, 100.0),
        rect_size=(60.0, 40.0),
    )
    original_copy = edge_image.copy()

    _ = extract_external_contours(edge_image)

    np.testing.assert_array_equal(edge_image, original_copy)


def test_degenerate_contours_rejected():
    """Verify degenerate contours (fewer than 3 points or zero side lengths) return None."""
    # 1-point contour
    cnt_1 = np.array([[[10, 10]]], dtype=np.int32)
    assert calculate_contour_geometry(cnt_1) is None

    # 2-point contour
    cnt_2 = np.array([[[10, 10]], [[20, 20]]], dtype=np.int32)
    assert calculate_contour_geometry(cnt_2) is None

    # Colinear points with 0 area
    cnt_colinear = np.array([[[10, 10]], [[20, 20]], [[30, 30]]], dtype=np.int32)
    assert calculate_contour_geometry(cnt_colinear) is None


def test_filter_contours_by_area():
    """Verify filter_contours_by_area helper retains only contours satisfying min_area."""
    # Create two valid contours
    cnt_large = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], dtype=np.int32)  # area 10000
    cnt_small = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)        # area 100

    contours = [cnt_large, cnt_small]

    filtered_low = filter_contours_by_area(contours, min_area=50.0)
    assert len(filtered_low) == 2

    filtered_high = filter_contours_by_area(contours, min_area=500.0)
    assert len(filtered_high) == 1
    assert filtered_high[0] is cnt_large


# ============================================================================
# Data Model and Helper Tests
# ============================================================================


def test_contour_geometry_aspect_ratio_and_measurement_result():
    """Verify aspect_ratio property and conversion to MeasurementResult."""
    contour = np.array([[[0, 0]], [[100, 0]], [[100, 50]], [[0, 50]]], dtype=np.int32)
    geom = calculate_contour_geometry(contour, min_area=0.0)

    assert geom is not None
    assert abs(geom.aspect_ratio - 2.0) < 0.1

    # Convert to MeasurementResult
    result = geom.to_measurement_result(
        object_id=1,
        length_cm=10.0,
        width_cm=5.0,
        pixels_per_cm=10.0,
    )

    assert isinstance(result, MeasurementResult)
    assert result.object_id == 1
    assert result.length_px == geom.length_px
    assert result.width_px == geom.width_px
    assert result.contour_area_px2 == geom.contour_area_px2
    assert result.length_cm == 10.0
    assert result.width_cm == 5.0
    assert result.pixels_per_cm == 10.0
    assert result.length_mm == 100.0
    assert result.width_mm == 50.0


# ============================================================================
# Input Validation and Error Tests
# ============================================================================


@pytest.mark.parametrize(
    "invalid_image,expected_error_substr",
    [
        (None, "NumPy array"),
        ("not_an_array", "NumPy array"),
        (np.array([]), "empty"),
        (np.zeros((100, 100, 3), dtype=np.uint8), "2D single-channel"),
        (np.zeros((100, 100), dtype=np.float32), "dtype uint8"),
    ],
)
def test_detect_contours_invalid_image(invalid_image, expected_error_substr):
    """Verify detect_contours clearly rejects invalid image arrays."""
    with pytest.raises(ContourDetectionError, match=expected_error_substr):
        detect_contours(invalid_image)


@pytest.mark.parametrize(
    "invalid_min_area",
    [
        -1.0,
        -50,
        "500",
        True,
        False,
        None,
    ],
)
def test_detect_contours_invalid_min_area(invalid_min_area):
    """Verify detect_contours rejects invalid min_area parameters."""
    valid_image = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ContourDetectionError):
        detect_contours(valid_image, min_area=invalid_min_area)


@pytest.mark.parametrize(
    "invalid_min_dimension",
    [
        -1.0,
        "1.0",
        True,
        False,
    ],
)
def test_detect_contours_invalid_min_dimension(invalid_min_dimension):
    """Verify detect_contours rejects invalid min_dimension parameters."""
    valid_image = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ContourDetectionError):
        detect_contours(valid_image, min_dimension=invalid_min_dimension)


def test_calculate_contour_geometry_invalid_contour():
    """Verify calculate_contour_geometry rejects non-array inputs."""
    with pytest.raises(ContourDetectionError, match="NumPy array"):
        calculate_contour_geometry(None)

    with pytest.raises(ContourDetectionError, match="NumPy array"):
        calculate_contour_geometry([[10, 10], [20, 20]])  # Python list instead of array

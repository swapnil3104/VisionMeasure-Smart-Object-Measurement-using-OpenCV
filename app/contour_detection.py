"""Contour detection and geometry module for VisionMeasure.

Provides functions to extract external object boundaries from edge maps,
filter tiny or degenerate contours, compute rotated minimum bounding rectangles,
and normalize pixel dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from models.measurement_result import MeasurementResult


class ContourDetectionError(ValueError):
    """Raised when an error occurs during contour extraction or geometry computation."""


@dataclass
class ContourGeometry:
    """Structured geometric metrics for a detected candidate contour.

    Attributes:
        contour: Raw contour point array as returned by OpenCV (shape: (N, 1, 2) or (N, 2)).
        contour_area_px2: Enclosed area of the contour in pixels squared (via cv2.contourArea).
        length_px: Longer side of the rotated bounding rectangle in pixels (normalized).
        width_px: Shorter side of the rotated bounding rectangle in pixels (normalized).
        center: (x, y) coordinates of the bounding rectangle centroid.
        angle_deg: Rotation angle in degrees of the bounding box returned by cv2.minAreaRect.
        box_points: 4 corner coordinates of the rotated rectangle (shape: (4, 2), dtype: float32).

    Note:
        These dimensions describe the rotated minimum bounding rectangle surrounding
        the candidate contour, which may differ from the object's true geometric dimensions
        for circular, concave, or non-rectangular shapes.
        Additionally, external contours represent candidate boundaries and must not be
        assumed to each represent a distinct physical object without further validation.
    """

    contour: np.ndarray
    contour_area_px2: float
    length_px: float
    width_px: float
    center: tuple[float, float]
    angle_deg: float
    box_points: np.ndarray

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of the bounding box (length_px / width_px)."""
        return self.length_px / self.width_px if self.width_px > 0.0 else 0.0

    def to_measurement_result(
        self,
        object_id: int,
        length_cm: Optional[float] = None,
        width_cm: Optional[float] = None,
        pixels_per_cm: Optional[float] = None,
    ) -> MeasurementResult:
        """Convert this contour geometry into a structured MeasurementResult record.

        Args:
            object_id: Sequential identifier for the measured object.
            length_cm: Optional estimated physical length in centimeters.
            width_cm: Optional estimated physical width in centimeters.
            pixels_per_cm: Optional calibration scale in pixels per centimeter.

        Returns:
            MeasurementResult instance populated with pixel and physical dimensions.
        """
        return MeasurementResult(
            object_id=object_id,
            length_px=self.length_px,
            width_px=self.width_px,
            contour_area_px2=self.contour_area_px2,
            length_cm=length_cm,
            width_cm=width_cm,
            pixels_per_cm=pixels_per_cm,
            box_points=np.copy(self.box_points) if self.box_points is not None else None,
            center=self.center,
        )


def _validate_edge_image(edge_image: np.ndarray) -> None:
    """Validate that the input is a valid 2D uint8 edge image array.

    Raises:
        ContourDetectionError: If the image array is invalid.
    """
    if edge_image is None or not isinstance(edge_image, np.ndarray):
        raise ContourDetectionError("Input edge image must be a valid NumPy array.")

    if edge_image.size == 0:
        raise ContourDetectionError("Input edge image array is empty.")

    if edge_image.ndim != 2:
        raise ContourDetectionError(
            f"Expected a 2D single-channel edge image, got shape {edge_image.shape}."
        )

    if edge_image.dtype != np.uint8:
        raise ContourDetectionError(
            f"Expected edge image array dtype uint8, got {edge_image.dtype}."
        )


def _validate_min_area(min_area: float) -> None:
    """Validate that min_area is a non-negative numeric value.

    Raises:
        ContourDetectionError: If min_area is invalid.
    """
    if isinstance(min_area, bool) or not isinstance(min_area, (int, float, np.number)):
        raise ContourDetectionError(
            f"Minimum contour area must be a numeric value, got {type(min_area).__name__}."
        )

    if min_area < 0:
        raise ContourDetectionError(
            f"Minimum contour area must be non-negative, got {min_area}."
        )


def _validate_min_dimension(min_dimension: float) -> None:
    """Validate that min_dimension is a non-negative numeric value.

    Raises:
        ContourDetectionError: If min_dimension is invalid.
    """
    if isinstance(min_dimension, bool) or not isinstance(min_dimension, (int, float, np.number)):
        raise ContourDetectionError(
            f"Minimum dimension must be a numeric value, got {type(min_dimension).__name__}."
        )

    if min_dimension < 0:
        raise ContourDetectionError(
            f"Minimum dimension must be non-negative, got {min_dimension}."
        )


def extract_external_contours(edge_image: np.ndarray) -> list[np.ndarray]:
    """Extract external boundary contours from a binary edge image.

    Uses cv2.RETR_EXTERNAL to ignore nested inner edges and retrieve only
    the outermost candidate boundaries.

    Args:
        edge_image: Single-channel 2D binary edge image array (dtype uint8).

    Returns:
        List of NumPy contour arrays.

    Raises:
        ContourDetectionError: If edge_image is invalid.
    """
    _validate_edge_image(edge_image)

    # Pass a copy to guarantee findContours does not mutate the source image
    contours_result = cv2.findContours(
        edge_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # OpenCV 4.x/5.x returns (contours, hierarchy), OpenCV 3.x returns (image, contours, hierarchy)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
    return list(contours)


def calculate_contour_geometry(
    contour: np.ndarray,
    min_area: float = 0.0,
    min_dimension: float = 1e-3,
) -> Optional[ContourGeometry]:
    """Calculate normalized rotated bounding box geometry for a single contour.

    Uses cv2.minAreaRect() to fit a minimum-area rotated rectangle around the
    contour points. Normalizes side lengths so that `length_px` is always the
    longer side and `width_px` is the shorter side. Filters out degenerate or
    near-zero side lengths.

    Args:
        contour: Single contour array (shape: (N, 1, 2) or (N, 2)).
        min_area: Minimum acceptable enclosed contour area in pixels squared.
        min_dimension: Minimum acceptable side length in pixels to reject degenerate lines/points.

    Returns:
        ContourGeometry instance if the contour is valid, or None if degenerate.

    Raises:
        ContourDetectionError: If contour is not a NumPy array or parameters are invalid.
    """
    if contour is None or not isinstance(contour, np.ndarray):
        raise ContourDetectionError("Contour must be a valid NumPy array.")

    _validate_min_area(min_area)
    _validate_min_dimension(min_dimension)

    # Degenerate contour check: a polygon requires at least 3 points
    if contour.shape[0] < 3:
        return None

    area = float(cv2.contourArea(contour))
    if area < min_area or area <= 0.0:
        return None

    # Calculate rotated bounding box: ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (dim_a, dim_b), angle_deg = rect

    side_a = float(dim_a)
    side_b = float(dim_b)

    # Normalize sides: length_px is longer, width_px is shorter
    length_px = max(side_a, side_b)
    width_px = min(side_a, side_b)

    # Reject zero or near-zero side lengths
    if length_px < min_dimension or width_px < min_dimension:
        return None

    # 4 corner coordinates (float32)
    box_points = cv2.boxPoints(rect).astype(np.float32)

    return ContourGeometry(
        contour=contour,
        contour_area_px2=area,
        length_px=length_px,
        width_px=width_px,
        center=(float(center_x), float(center_y)),
        angle_deg=float(angle_deg),
        box_points=box_points,
    )


def filter_contours_by_area(
    contours: list[np.ndarray],
    min_area: float = 500.0,
) -> list[np.ndarray]:
    """Filter raw contours based on a minimum enclosed area threshold.

    Args:
        contours: List of raw contour NumPy arrays.
        min_area: Minimum contour area in pixels squared. Must be non-negative.

    Returns:
        Filtered list of contours whose cv2.contourArea is >= min_area.

    Raises:
        ContourDetectionError: If min_area is invalid.
    """
    _validate_min_area(min_area)

    filtered: list[np.ndarray] = []
    for cnt in contours:
        if cnt is not None and isinstance(cnt, np.ndarray) and cnt.shape[0] >= 3:
            if float(cv2.contourArea(cnt)) >= min_area:
                filtered.append(cnt)
    return filtered


def detect_contours(
    edge_image: np.ndarray,
    min_area: float = 500.0,
    min_dimension: float = 1.0,
) -> list[ContourGeometry]:
    """Extract external contours and compute rotated bounding rectangles.

    Full pipeline for Phase 3:
    1. Validates input edge image array and configuration parameters.
    2. Extracts external contours.
    3. Filters contours by min_area and excludes degenerate boundaries.
    4. Computes rotated bounding rectangles using cv2.minAreaRect().
    5. Normalizes side lengths so length_px >= width_px.
    6. Returns structured ContourGeometry records (no in-place image drawing).

    Args:
        edge_image: Single-channel 2D binary edge image array (dtype uint8).
        min_area: Minimum acceptable enclosed contour area in pixels squared.
        min_dimension: Minimum acceptable side length in pixels.

    Returns:
        List of structured ContourGeometry instances for accepted candidate objects.
        Returns an empty list without error if no valid contours are found.

    Raises:
        ContourDetectionError: If inputs or threshold parameters are invalid.
    """
    _validate_edge_image(edge_image)
    _validate_min_area(min_area)
    _validate_min_dimension(min_dimension)

    raw_contours = extract_external_contours(edge_image)
    if not raw_contours:
        return []

    geometries: list[ContourGeometry] = []
    for contour in raw_contours:
        geometry = calculate_contour_geometry(
            contour,
            min_area=min_area,
            min_dimension=min_dimension,
        )
        if geometry is not None:
            geometries.append(geometry)

    return geometries

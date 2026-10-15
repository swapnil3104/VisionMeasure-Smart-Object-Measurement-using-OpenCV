"""Measurement engine for VisionMeasure.

Converts pixel dimensions of detected bounding rectangles into calibrated
physical dimensions (centimeters and millimeters) while preserving original
pixel-space measurements and explicitly handling uncalibrated states.

Planar Constraint Requirement:
    Physical dimension calculations rely on a single scale factor (pixels_per_unit).
    This conversion is mathematically sound ONLY if the target object lies on the
    same plane as the reference object used during calibration, and if the camera's
    optical axis is perpendicular to that plane. Height differences, tilting, or
    perspective differences will introduce dimensional errors.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

from app.calibration import CalibrationData
from models.measurement_result import MeasurementResult

if TYPE_CHECKING:
    from app.contour_detection import ContourGeometry


class MeasurementError(ValueError):
    """Raised when measurement calculations encounter invalid geometry or scale."""


def _validate_numeric(value: Union[int, float], name: str) -> float:
    """Validate that value is a finite float and not boolean, NaN, or infinite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{name} must be a number, got {type(value).__name__}.")
    val_float = float(value)
    if math.isnan(val_float) or math.isinf(val_float):
        raise MeasurementError(f"{name} must be finite, got {val_float}.")
    return val_float


def calculate_physical_dimension(
    pixel_length: Union[int, float],
    pixels_per_unit: Union[int, float],
) -> float:
    """Convert a pixel measurement to a physical dimension using a scale factor.

    Formula:
        physical_dimension = length_in_pixels / pixels_per_unit

    Args:
        pixel_length: Length in pixels (must be non-negative).
        pixels_per_unit: Scale factor in pixels per physical unit (must be > 0).

    Returns:
        Calculated physical dimension.

    Raises:
        MeasurementError: If pixel_length is negative or pixels_per_unit is non-positive.
    """
    px = _validate_numeric(pixel_length, "Pixel length")
    scale = _validate_numeric(pixels_per_unit, "Pixels per unit")

    if px < 0.0:
        raise MeasurementError(f"Pixel length cannot be negative, got {px}.")
    if scale <= 0.0:
        raise MeasurementError(f"Scale factor (pixels_per_unit) must be strictly positive, got {scale}.")

    return px / scale


def validate_geometry(geometry: "ContourGeometry") -> None:
    """Validate that contour geometry is non-degenerate and physically meaningful.

    Args:
        geometry: Candidate contour geometry.

    Raises:
        MeasurementError: If geometry is None, has non-positive sides, negative area, or NaN/Inf.
    """
    if geometry is None:
        raise MeasurementError("Contour geometry cannot be None.")

    length_px = _validate_numeric(geometry.length_px, "geometry.length_px")
    width_px = _validate_numeric(geometry.width_px, "geometry.width_px")
    area = _validate_numeric(geometry.contour_area_px2, "geometry.contour_area_px2")

    if length_px <= 0.0 or width_px <= 0.0:
        raise MeasurementError(
            f"Degenerate geometry: bounding dimensions must be strictly positive. "
            f"Got length_px={length_px}, width_px={width_px}."
        )

    if area < 0.0:
        raise MeasurementError(f"Invalid geometry: contour area cannot be negative. Got {area}.")


def measure_object(
    geometry: "ContourGeometry",
    object_id: int,
    calibration: Optional[CalibrationData] = None,
) -> MeasurementResult:
    """Compute measurements for a single detected object bounding box.

    Converts pixel dimensions to centimeters and millimeters if valid calibration
    is supplied; otherwise, leaves physical dimension fields as None (explicitly uncalibrated).
    Raw pixel dimensions are always preserved.

    Args:
        geometry: Validated ContourGeometry of the detected object.
        object_id: Sequential identifier for the object (must be >= 0).
        calibration: Optional CalibrationData. If None, physical measurements are None.

    Returns:
        MeasurementResult containing pixel metrics and optional physical measurements.

    Raises:
        MeasurementError: If geometry is degenerate or calibration instance is invalid.
    """
    if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id < 0:
        raise MeasurementError(f"Object ID must be a non-negative integer, got {object_id}.")

    validate_geometry(geometry)

    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    pixels_per_cm: Optional[float] = None

    if calibration is not None:
        if not isinstance(calibration, CalibrationData):
            raise MeasurementError(
                f"Expected calibration to be CalibrationData instance, got {type(calibration).__name__}."
            )
        if calibration.pixels_per_cm <= 0.0:
            raise MeasurementError("Calibration data has non-positive pixels_per_cm.")

        length_cm = calculate_physical_dimension(geometry.length_px, calibration.pixels_per_cm)
        width_cm = calculate_physical_dimension(geometry.width_px, calibration.pixels_per_cm)
        pixels_per_cm = calibration.pixels_per_cm

    return MeasurementResult(
        object_id=object_id,
        length_px=float(geometry.length_px),
        width_px=float(geometry.width_px),
        contour_area_px2=float(geometry.contour_area_px2),
        length_cm=length_cm,
        width_cm=width_cm,
        pixels_per_cm=pixels_per_cm,
        box_points=np.copy(geometry.box_points) if geometry.box_points is not None else None,
        center=geometry.center,
    )


def measure_objects(
    geometries: Sequence["ContourGeometry"],
    calibration: Optional[CalibrationData] = None,
    start_id: int = 1,
) -> List[MeasurementResult]:
    """Measure a sequence of candidate object contours.

    Args:
        geometries: Sequence of ContourGeometry items.
        calibration: Optional CalibrationData applied to all objects.
        start_id: Starting index for sequential object IDs (default: 1).

    Returns:
        List of MeasurementResult instances. Returns empty list if geometries is empty.

    Raises:
        MeasurementError: If start_id is invalid or any geometry is degenerate.
    """
    if isinstance(start_id, bool) or not isinstance(start_id, int) or start_id < 0:
        raise MeasurementError(f"start_id must be a non-negative integer, got {start_id}.")

    results: List[MeasurementResult] = []
    for idx, geom in enumerate(geometries):
        res = measure_object(
            geometry=geom,
            object_id=start_id + idx,
            calibration=calibration,
        )
        results.append(res)
    return results

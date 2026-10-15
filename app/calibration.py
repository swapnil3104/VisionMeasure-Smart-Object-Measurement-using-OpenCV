"""Calibration engine for VisionMeasure.

Calculates the scale factor (pixels_per_unit) using a known reference object,
validates calibration parameters, and supports consistent unit conversions
between centimeters and millimeters.

Planar Constraint Requirement:
    Computer vision dimension measurements from a 2D camera image assume an
    orthographic or paraxial projection onto a flat reference plane.
    The calibration scale factor calculated here is physically valid ONLY when
    both the known reference object and the target objects to be measured lie
    on approximately the same working plane and are viewed perpendicularly
    to the camera's optical axis. Objects located closer to or further from
    the camera lens, or tilted relative to the imaging plane, will exhibit
    perspective foreshortening and scale variations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from app.contour_detection import ContourGeometry


class CalibrationError(ValueError):
    """Raised when an error occurs during scale calculation or validation."""


SUPPORTED_UNITS = ("cm", "mm")


@dataclass(frozen=True)
class CalibrationData:
    """Immutable representation of a validated camera scale calibration.

    Attributes:
        pixels_per_cm: Scale factor expressed in pixels per centimeter.
        reference_px: The pixel dimension of the reference object used.
        reference_length: The known real-world dimension of the reference object.
        unit: Real-world unit used during calibration ('cm' or 'mm').
        pixels_per_unit: Scale factor in pixels per the specified unit.

    Planar Constraint Note:
        This calibration is valid only for objects on the same working plane
        as the reference object, perpendicular to the camera optical axis.
    """

    pixels_per_cm: float
    reference_px: float
    reference_length: float
    unit: str
    pixels_per_unit: float

    @property
    def pixels_per_mm(self) -> float:
        """Scale factor expressed in pixels per millimeter."""
        return self.pixels_per_cm / 10.0

    def pixels_to_cm(self, pixels: float) -> float:
        """Convert a pixel span to centimeters.

        Args:
            pixels: Pixel span to convert (must be non-negative).

        Returns:
            Calculated length in centimeters.

        Raises:
            CalibrationError: If pixels is negative or non-numeric.
        """
        _validate_dimension_pixels(pixels)
        return float(pixels) / self.pixels_per_cm

    def pixels_to_mm(self, pixels: float) -> float:
        """Convert a pixel span to millimeters.

        Args:
            pixels: Pixel span to convert (must be non-negative).

        Returns:
            Calculated length in millimeters.

        Raises:
            CalibrationError: If pixels is negative or non-numeric.
        """
        _validate_dimension_pixels(pixels)
        return float(pixels) / self.pixels_per_mm

    def pixels_to_unit(self, pixels: float) -> float:
        """Convert a pixel span to the unit configured during calibration.

        Args:
            pixels: Pixel span to convert (must be non-negative).

        Returns:
            Calculated length in the configured reference unit.

        Raises:
            CalibrationError: If pixels is negative or non-numeric.
        """
        _validate_dimension_pixels(pixels)
        return float(pixels) / self.pixels_per_unit


def _validate_numeric(value: Union[int, float], name: str) -> float:
    """Ensure value is a valid, finite numeric float (rejecting booleans, NaN, Inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{name} must be a valid number, got {type(value).__name__}.")
    val_float = float(value)
    if math.isnan(val_float) or math.isinf(val_float):
        raise CalibrationError(f"{name} must be a finite number, got {val_float}.")
    return val_float


def _validate_dimension_pixels(pixels: Union[int, float]) -> float:
    """Validate pixel span for conversion (must be non-negative and finite)."""
    val = _validate_numeric(pixels, "Pixel value")
    if val < 0.0:
        raise CalibrationError(f"Pixel value cannot be negative, got {val}.")
    return val


def normalize_unit(unit: str) -> str:
    """Normalize and validate the physical measurement unit.

    Args:
        unit: Unit string (e.g. 'cm', 'CM', 'mm', 'MM').

    Returns:
        Normalized lowercase unit string ('cm' or 'mm').

    Raises:
        CalibrationError: If unit is unsupported or empty.
    """
    if not isinstance(unit, str):
        raise CalibrationError(f"Unit must be a string, got {type(unit).__name__}.")
    cleaned = unit.strip().lower()
    if cleaned not in SUPPORTED_UNITS:
        raise CalibrationError(
            f"Unsupported unit '{unit}'. Supported units are {SUPPORTED_UNITS}."
        )
    return cleaned


def calculate_pixels_per_unit(
    reference_px: Union[int, float],
    reference_length: Union[int, float],
    unit: str = "cm",
) -> float:
    """Calculate the scale factor in pixels per specified unit.

    Formula:
        pixels_per_unit = reference_pixels / known_reference_length

    Args:
        reference_px: Reference object span in pixels (must be > 0).
        reference_length: Known physical dimension of the reference object (must be > 0).
        unit: Measurement unit of the reference length ('cm' or 'mm').

    Returns:
        Scale factor in pixels per specified unit.

    Raises:
        CalibrationError: If inputs are non-numeric, <= 0, or unsupported unit.
    """
    px = _validate_numeric(reference_px, "Reference pixel span")
    length = _validate_numeric(reference_length, "Reference length")
    normalize_unit(unit)

    if px <= 0.0:
        raise CalibrationError(f"Reference pixel span must be strictly positive, got {px}.")
    if length <= 0.0:
        raise CalibrationError(f"Reference length must be strictly positive, got {length}.")

    return px / length


def calibrate(
    reference_px: Union[int, float],
    reference_length: Union[int, float],
    unit: str = "cm",
) -> CalibrationData:
    """Calibrate the vision system using a known reference object.

    Args:
        reference_px: Reference object span in pixels (must be > 0).
        reference_length: Known physical dimension of the reference object (must be > 0).
        unit: Measurement unit of the reference length ('cm' or 'mm').

    Returns:
        CalibrationData instance containing verified scales for cm and mm.

    Raises:
        CalibrationError: If inputs are invalid, non-positive, or unsupported.
    """
    norm_unit = normalize_unit(unit)
    pixels_per_unit = calculate_pixels_per_unit(reference_px, reference_length, norm_unit)

    if norm_unit == "cm":
        pixels_per_cm = pixels_per_unit
    elif norm_unit == "mm":
        # If 10 mm = 1 cm, then pixels_per_cm = pixels_per_mm * 10
        pixels_per_cm = pixels_per_unit * 10.0
    else:  # pragma: no cover
        raise CalibrationError(f"Unhandled unit '{norm_unit}'.")

    return CalibrationData(
        pixels_per_cm=pixels_per_cm,
        reference_px=float(reference_px),
        reference_length=float(reference_length),
        unit=norm_unit,
        pixels_per_unit=pixels_per_unit,
    )


def calibrate_from_geometry(
    geometry: "ContourGeometry",
    reference_length: Union[int, float],
    unit: str = "cm",
    side: str = "longer",
) -> CalibrationData:
    """Calibrate using the detected geometry of a known reference object.

    Args:
        geometry: Detected ContourGeometry of the reference object.
        reference_length: Known physical length of the reference object.
        unit: Physical unit ('cm' or 'mm').
        side: Bounding rectangle side to calibrate against:
            - 'longer' or 'length': uses geometry.length_px
            - 'shorter' or 'width': uses geometry.width_px

    Returns:
        CalibrationData instance.

    Raises:
        CalibrationError: If geometry is invalid or side is unrecognized.
    """
    if geometry is None:
        raise CalibrationError("Reference geometry cannot be None.")

    side_clean = side.strip().lower()
    if side_clean in ("longer", "length"):
        pixel_span = geometry.length_px
    elif side_clean in ("shorter", "width"):
        pixel_span = geometry.width_px
    else:
        raise CalibrationError(
            f"Invalid reference side '{side}'. Expected 'longer', 'length', 'shorter', or 'width'."
        )

    return calibrate(
        reference_px=pixel_span,
        reference_length=reference_length,
        unit=unit,
    )


def calculate_pixel_distance(
    point1: Tuple[Union[int, float], Union[int, float]],
    point2: Tuple[Union[int, float], Union[int, float]],
) -> float:
    """Calculate Euclidean distance between two 2D points in pixel coordinates.

    Args:
        point1: (x1, y1) coordinate pair.
        point2: (x2, y2) coordinate pair.

    Returns:
        Distance in pixels.

    Raises:
        CalibrationError: If points are not valid 2D coordinates or distance is zero.
    """
    try:
        x1, y1 = point1
        x2, y2 = point2
    except (TypeError, ValueError) as e:
        raise CalibrationError(f"Points must be 2-tuples of numbers, got {point1}, {point2}.") from e

    x1_val = _validate_numeric(x1, "Point 1 X")
    y1_val = _validate_numeric(y1, "Point 1 Y")
    x2_val = _validate_numeric(x2, "Point 2 X")
    y2_val = _validate_numeric(y2, "Point 2 Y")

    dist = math.hypot(x2_val - x1_val, y2_val - y1_val)
    if dist <= 0.0:
        raise CalibrationError("Distance between calibration points cannot be zero.")
    return dist


def calibrate_from_points(
    point1: Tuple[Union[int, float], Union[int, float]],
    point2: Tuple[Union[int, float], Union[int, float]],
    reference_length: Union[int, float],
    unit: str = "cm",
) -> CalibrationData:
    """Calibrate by measuring pixel distance between two user-selected points.

    Useful when reference markers, ruler markings, or coin diameters are marked
    interactively via GUI point clicks.

    Args:
        point1: (x1, y1) starting point in pixels.
        point2: (x2, y2) ending point in pixels.
        reference_length: Known real-world distance between the points.
        unit: Physical unit ('cm' or 'mm').

    Returns:
        CalibrationData instance.
    """
    dist_px = calculate_pixel_distance(point1, point2)
    return calibrate(
        reference_px=dist_px,
        reference_length=reference_length,
        unit=unit,
    )

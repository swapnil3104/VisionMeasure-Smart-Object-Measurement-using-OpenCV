"""Data model representing object measurement results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MeasurementResult:
    """Represents the measurement metrics for a detected object bounding box.

    Attributes:
        object_id: Sequential identifier for the detected object.
        length_px: Longest side of the bounding rectangle in pixels.
        width_px: Shorter side of the bounding rectangle in pixels.
        contour_area_px2: Area enclosed by the contour in pixels squared.
        length_cm: Estimated physical length in centimeters (or None if uncalibrated).
        width_cm: Estimated physical width in centimeters (or None if uncalibrated).
        pixels_per_cm: Calibration scale used (pixels per centimeter).
    """

    object_id: int
    length_px: float
    width_px: float
    contour_area_px2: float
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    pixels_per_cm: Optional[float] = None
    box_points: Optional[np.ndarray] = None
    center: Optional[tuple[float, float]] = None

    @property
    def is_calibrated(self) -> bool:
        """Indicate whether physical calibration was applied."""
        return self.length_cm is not None and self.width_cm is not None and self.pixels_per_cm is not None

    @property
    def pixels_per_mm(self) -> Optional[float]:
        """Calibration scale in pixels per millimeter."""
        return self.pixels_per_cm / 10.0 if self.pixels_per_cm is not None else None

    @property
    def length_mm(self) -> Optional[float]:
        """Estimated physical length in millimeters."""
        return self.length_cm * 10.0 if self.length_cm is not None else None

    @property
    def width_mm(self) -> Optional[float]:
        """Estimated physical width in millimeters."""
        return self.width_cm * 10.0 if self.width_cm is not None else None

    def to_dict(self) -> dict:
        """Return a structured dictionary representation of the measurement result."""
        return {
            "object_id": self.object_id,
            "length_px": round(self.length_px, 2),
            "width_px": round(self.width_px, 2),
            "contour_area_px2": round(self.contour_area_px2, 2),
            "is_calibrated": self.is_calibrated,
            "length_cm": round(self.length_cm, 3) if self.length_cm is not None else None,
            "width_cm": round(self.width_cm, 3) if self.width_cm is not None else None,
            "length_mm": round(self.length_mm, 2) if self.length_mm is not None else None,
            "width_mm": round(self.width_mm, 2) if self.width_mm is not None else None,
            "pixels_per_cm": round(self.pixels_per_cm, 3) if self.pixels_per_cm is not None else None,
        }

    def format_dimension_string(self, unit: str = "cm", include_id: bool = True) -> str:
        """Format dimensions into a human-readable display string.

        Args:
            unit: Preferred physical unit ('cm' or 'mm'). Ignored if uncalibrated.
            include_id: Whether to prefix the string with 'Obj {id}: '.

        Returns:
            Human-readable formatted measurement string. Physical units are strictly
            omitted if the result is uncalibrated.
        """
        prefix = f"Obj {self.object_id}: " if include_id else ""
        norm_unit = unit.strip().lower()

        if self.is_calibrated and self.length_cm is not None and self.width_cm is not None:
            if norm_unit in ("mm", "millimeter", "millimeters"):
                l_val = self.length_mm if self.length_mm is not None else self.length_cm * 10.0
                w_val = self.width_mm if self.width_mm is not None else self.width_cm * 10.0
                return f"{prefix}{l_val:.1f} x {w_val:.1f} mm"
            return f"{prefix}{self.length_cm:.2f} x {self.width_cm:.2f} cm"

        # Explicit uncalibrated representation - never output physical units
        return f"{prefix}{self.length_px:.1f} x {self.width_px:.1f} px (Uncalibrated)"


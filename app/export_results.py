"""Results export module for VisionMeasure.

Provides functions to save annotated images and export measurement metrics
to CSV files with comprehensive validation and error handling.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np

from app.image_input import validate_image_array
from models.measurement_result import MeasurementResult


class ExportError(Exception):
    """Base exception for all export-related errors."""


class ImageExportError(ExportError, IOError):
    """Raised when an annotated image cannot be saved or encoded."""


class CSVExportError(ExportError, IOError):
    """Raised when measurement results cannot be exported to CSV."""


CSV_HEADERS = [
    "object_id",
    "length_px",
    "width_px",
    "contour_area_px2",
    "length_cm",
    "width_cm",
    "pixels_per_cm",
    "source_image",
]

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}


def save_annotated_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    overwrite: bool = True,
) -> str:
    """Save an annotated image to disk and verify that it was written successfully.

    Args:
        image: Annotated BGR image array (H, W, 3).
        output_path: Destination file path (supports .jpg, .jpeg, .png).
        overwrite: If False, raises ImageExportError if file already exists.

    Returns:
        Canonical string path to the saved file.

    Raises:
        ImageExportError: If image array is invalid, path is invalid, format is unsupported,
            or the file write operation fails.
    """
    try:
        validate_image_array(image, "annotated output")
    except Exception as exc:
        raise ImageExportError(f"Cannot export invalid image: {exc}") from exc

    if not output_path:
        raise ImageExportError("Output image path cannot be empty.")

    dest = Path(output_path).resolve()
    ext = dest.suffix.lower()

    if ext not in SUPPORTED_IMAGE_FORMATS:
        supported_str = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise ImageExportError(
            f"Unsupported image export extension '{ext}'. Supported formats: {supported_str}"
        )

    if dest.exists() and not overwrite:
        raise ImageExportError(f"Destination file already exists and overwrite is False: {dest}")

    # Ensure parent directory exists
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ImageExportError(f"Failed to create directory '{dest.parent}': {exc}") from exc

    # Attempt save via cv2.imencode for reliable cross-platform and Unicode path handling
    try:
        success, encoded_img = cv2.imencode(ext, image)
        if not success or encoded_img is None:
            raise ImageExportError(f"OpenCV failed to encode image format '{ext}'.")

        with open(dest, "wb") as f:
            encoded_img.tofile(f)
    except ImageExportError:
        raise
    except Exception as exc:
        raise ImageExportError(f"Failed to write image to '{dest}': {exc}") from exc

    if not dest.is_file() or dest.stat().st_size == 0:
        raise ImageExportError(f"Image write verification failed for '{dest}'.")

    return str(dest)


def export_csv(
    measurements: Sequence[MeasurementResult],
    output_path: Union[str, Path],
    source_image: str = "",
    overwrite: bool = True,
) -> str:
    """Export measurement results to a structured CSV file.

    Columns:
        object_id, length_px, width_px, contour_area_px2, length_cm, width_cm,
        pixels_per_cm, source_image

    Args:
        measurements: Sequence of MeasurementResult records.
        output_path: Destination path for the CSV file.
        source_image: Optional filename or label of the source image.
        overwrite: If False, raises CSVExportError if file already exists.

    Returns:
        Canonical string path to the exported CSV file.

    Raises:
        CSVExportError: If path is invalid, file cannot be written, or measurement objects are invalid.
    """
    if not output_path:
        raise CSVExportError("CSV output path cannot be empty.")

    dest = Path(output_path).resolve()
    if dest.suffix.lower() != ".csv":
        dest = dest.with_suffix(".csv")

    if dest.exists() and not overwrite:
        raise CSVExportError(f"Destination file already exists and overwrite is False: {dest}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise CSVExportError(f"Failed to create directory '{dest.parent}': {exc}") from exc

    # Validate items
    rows: list[list[Union[str, int, float]]] = []
    source_name = Path(source_image).name if source_image else "N/A"

    for idx, item in enumerate(measurements):
        if not isinstance(item, MeasurementResult):
            raise CSVExportError(
                f"Item at index {idx} is not a MeasurementResult instance, got {type(item).__name__}."
            )

        row = [
            item.object_id,
            round(item.length_px, 2),
            round(item.width_px, 2),
            round(item.contour_area_px2, 2),
            round(item.length_cm, 3) if item.length_cm is not None else "",
            round(item.width_cm, 3) if item.width_cm is not None else "",
            round(item.pixels_per_cm, 3) if item.pixels_per_cm is not None else "",
            source_name,
        ]
        rows.append(row)

    try:
        with open(dest, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            writer.writerows(rows)
    except Exception as exc:
        raise CSVExportError(f"Failed to write CSV to '{dest}': {exc}") from exc

    if not dest.is_file():
        raise CSVExportError(f"CSV file verification failed for '{dest}'.")

    return str(dest)

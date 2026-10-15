"""Image input and validation module for VisionMeasure.

Provides functions to safely load still images and capture webcam frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


class ImageInputError(Exception):
    """Base exception for all image input errors."""


class ImageNotFoundError(ImageInputError, FileNotFoundError):
    """Raised when the specified image file does not exist."""


class UnsupportedFormatError(ImageInputError, ValueError):
    """Raised when the file format or extension is not supported."""


class ImageReadError(ImageInputError, IOError):
    """Raised when the image file cannot be read or decoded."""


class CameraError(ImageInputError, RuntimeError):
    """Raised when the camera cannot be opened or fails to capture a frame."""


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_array(image: np.ndarray, source_name: str = "image") -> None:
    """Validate that the given array is a non-empty 3-channel BGR image.

    Args:
        image: NumPy array to validate.
        source_name: Descriptive name for error messages.

    Raises:
        ImageReadError: If the array is invalid, empty, or not 3-channel BGR.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ImageReadError(f"Invalid image data received from {source_name}: not a valid NumPy array.")

    if image.size == 0:
        raise ImageReadError(f"Invalid image data received from {source_name}: array is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageReadError(
            f"Invalid image format from {source_name}: expected 3-channel BGR image, "
            f"got shape {image.shape}."
        )

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ImageReadError(f"Invalid dimensions {width}x{height} for {source_name}.")


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from a file path and return it as a BGR NumPy array.

    Args:
        path: Path to the image file (string or Path object).

    Returns:
        BGR image array of shape (height, width, 3).

    Raises:
        ImageNotFoundError: If the file does not exist.
        UnsupportedFormatError: If the file extension is not supported.
        ImageReadError: If the image cannot be read or decoded.
    """
    if not path:
        raise ImageNotFoundError("Image path must not be empty.")

    file_path = Path(path).resolve()

    if not file_path.is_file():
        raise ImageNotFoundError(f"Image file does not exist: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext}'. Supported formats: {supported_str}"
        )

    # Use cv2.imread; if it fails (e.g. unicode path on Windows), fall back to cv2.imdecode
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        try:
            image_data = np.fromfile(str(file_path), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise ImageReadError(f"Failed to read image from '{file_path}': {exc}") from exc

    if image is None:
        raise ImageReadError(f"Failed to decode image from '{file_path}'. File may be corrupt.")

    validate_image_array(image, str(file_path))
    return image


def capture_frame(camera_index: int = 0) -> np.ndarray:
    """Capture a single frame from the specified webcam index.

    Args:
        camera_index: Index of the camera device (default is 0).

    Returns:
        Captured BGR image array of shape (height, width, 3).

    Raises:
        CameraError: If the webcam cannot be opened or fails to yield a valid frame.
    """
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise CameraError(
                f"Cannot open camera index {camera_index}. Ensure the webcam is connected, "
                "turned on, and not currently used by another application."
            )

        ret, frame = cap.read()
        if not ret or frame is None:
            raise CameraError(
                f"Failed to capture a valid frame from camera index {camera_index}."
            )

        validate_image_array(frame, f"webcam index {camera_index}")
        return frame
    finally:
        cap.release()

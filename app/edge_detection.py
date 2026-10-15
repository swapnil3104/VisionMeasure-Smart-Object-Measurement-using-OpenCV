"""Edge detection module for VisionMeasure.

Provides functions to perform Canny edge detection and optional morphological
closing to clean and connect edge boundaries.
"""
from __future__ import annotations

import cv2
import numpy as np


class EdgeDetectionError(ValueError):
    """Raised when an error occurs during edge detection or invalid parameters are provided."""


def apply_morphological_closing(
    edge_image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply morphological closing (dilation followed by erosion) to close small gaps in edges.

    Args:
        edge_image: Single-channel binary/edge image array (shape: H, W).
        kernel_size: Size of the rectangular structuring element. Must be a positive odd integer.

    Returns:
        Processed binary edge image array with the same shape and dtype.

    Raises:
        EdgeDetectionError: If edge_image is invalid or kernel_size is not a positive odd integer.
    """
    if edge_image is None or not isinstance(edge_image, np.ndarray):
        raise EdgeDetectionError("Input edge image must be a valid NumPy array.")

    if edge_image.size == 0:
        raise EdgeDetectionError("Input edge image array is empty.")

    if edge_image.ndim != 2:
        raise EdgeDetectionError(
            f"Expected a 2D single-channel edge image, got shape {edge_image.shape}."
        )

    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise EdgeDetectionError(
            f"Morphology kernel size must be an integer, got {type(kernel_size).__name__}."
        )

    if kernel_size <= 0:
        raise EdgeDetectionError(
            f"Morphology kernel size must be positive, got {kernel_size}."
        )

    if kernel_size % 2 == 0:
        raise EdgeDetectionError(
            f"Morphology kernel size must be an odd number, got {kernel_size}."
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (int(kernel_size), int(kernel_size)),
    )
    return cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel)


def detect_edges(
    gray_image: np.ndarray,
    threshold1: int = 50,
    threshold2: int = 150,
    apply_closing: bool = False,
    closing_kernel_size: int = 3,
) -> np.ndarray:
    """Detect edges in a grayscale image using Canny edge detection.

    Args:
        gray_image: Single-channel 2D grayscale NumPy array (shape: H, W).
        threshold1: Lower hysteresis threshold for Canny edge detection (must be >= 0).
        threshold2: Upper hysteresis threshold for Canny edge detection (must be > threshold1).
        apply_closing: If True, applies morphological closing after Canny to bridge edge gaps.
        closing_kernel_size: Structuring element size for morphological closing (positive odd integer).

    Returns:
        Binary edge map of shape (H, W) and dtype uint8 where edges are 255 and background is 0.

    Raises:
        EdgeDetectionError: If inputs or threshold parameters are invalid.
    """
    if gray_image is None or not isinstance(gray_image, np.ndarray):
        raise EdgeDetectionError("Input grayscale image must be a valid NumPy array.")

    if gray_image.size == 0:
        raise EdgeDetectionError("Input grayscale image array is empty.")

    if gray_image.ndim != 2:
        raise EdgeDetectionError(
            f"Expected a 2D single-channel grayscale image, got shape {gray_image.shape}."
        )

    if gray_image.dtype != np.uint8:
        raise EdgeDetectionError(
            f"Expected grayscale image array dtype uint8, got {gray_image.dtype}."
        )

    # Validate threshold types (reject booleans and non-numeric types)
    if isinstance(threshold1, bool) or not isinstance(threshold1, (int, float, np.number)):
        raise EdgeDetectionError(
            f"Lower threshold must be a number, got {type(threshold1).__name__}."
        )

    if isinstance(threshold2, bool) or not isinstance(threshold2, (int, float, np.number)):
        raise EdgeDetectionError(
            f"Upper threshold must be a number, got {type(threshold2).__name__}."
        )

    if threshold1 < 0:
        raise EdgeDetectionError(
            f"Lower threshold must be >= 0, got {threshold1}."
        )

    if threshold1 >= threshold2:
        raise EdgeDetectionError(
            f"Lower threshold ({threshold1}) must be strictly less than upper threshold ({threshold2})."
        )

    edges = cv2.Canny(gray_image, threshold1=int(threshold1), threshold2=int(threshold2))

    if apply_closing:
        edges = apply_morphological_closing(edges, kernel_size=closing_kernel_size)

    return edges

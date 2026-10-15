"""Image preprocessing module for VisionMeasure.

Provides functions to convert images to grayscale and apply Gaussian blur
for noise reduction prior to edge detection.
"""
from __future__ import annotations

import cv2
import numpy as np


class PreprocessingError(ValueError):
    """Raised when an error occurs during image preprocessing or invalid arguments are given."""


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a 3-channel BGR image to single-channel grayscale.

    Args:
        image: NumPy array representing a 3-channel BGR image (shape: H, W, 3).

    Returns:
        Single-channel 2D grayscale image array (shape: H, W, dtype: uint8).

    Raises:
        PreprocessingError: If the input is not a valid 3-channel BGR array.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise PreprocessingError("Input image must be a valid NumPy array.")

    if image.size == 0:
        raise PreprocessingError("Input image array is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError(
            f"Expected a 3-channel BGR image, got shape {image.shape}."
        )

    if image.dtype != np.uint8:
        raise PreprocessingError(
            f"Expected image array dtype uint8, got {image.dtype}."
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma_x: float = 0.0,
) -> np.ndarray:
    """Apply Gaussian blur to an image array to reduce noise.

    Args:
        image: NumPy array (grayscale 2D or color 3D).
        kernel_size: Size of the Gaussian kernel. Must be a positive odd integer.
        sigma_x: Gaussian kernel standard deviation in X direction. Default is 0.0 (calculated from kernel size).

    Returns:
        Blurred image array with the same shape and dtype as the input.

    Raises:
        PreprocessingError: If input is invalid or kernel_size is not a positive odd integer.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise PreprocessingError("Input image must be a valid NumPy array.")

    if image.size == 0:
        raise PreprocessingError("Input image array is empty.")

    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise PreprocessingError(
            f"Blur kernel size must be an integer, got {type(kernel_size).__name__}."
        )

    if kernel_size <= 0:
        raise PreprocessingError(
            f"Blur kernel size must be positive, got {kernel_size}."
        )

    if kernel_size % 2 == 0:
        raise PreprocessingError(
            f"Blur kernel size must be an odd number, got {kernel_size}."
        )

    return cv2.GaussianBlur(
        image,
        (int(kernel_size), int(kernel_size)),
        sigmaX=float(sigma_x),
    )


def preprocess_image(image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
    """Preprocess a BGR image by converting to grayscale and applying Gaussian blur.

    Args:
        image: 3-channel BGR image array.
        blur_kernel: Size of Gaussian blur kernel (positive odd integer, default 5).

    Returns:
        Preprocessed single-channel 2D grayscale blurred image.

    Raises:
        PreprocessingError: If input image or blur_kernel is invalid.
    """
    gray = to_grayscale(image)
    return apply_gaussian_blur(gray, kernel_size=blur_kernel)

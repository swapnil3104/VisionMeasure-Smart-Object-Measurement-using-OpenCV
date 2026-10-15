"""Tests for preprocessing and edge detection modules (Phase 2)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.edge_detection import (
    EdgeDetectionError,
    apply_morphological_closing,
    detect_edges,
)
from app.image_input import load_image
from app.preprocessing import (
    PreprocessingError,
    apply_gaussian_blur,
    preprocess_image,
    to_grayscale,
)


def create_synthetic_test_image(height: int = 300, width: int = 400) -> np.ndarray:
    """Create a deterministic, reproducible BGR synthetic image for testing.

    Draws high-contrast geometric objects (a bright rectangle and a dark rectangle)
    on a neutral background.

    Args:
        height: Height of the generated image.
        width: Width of the generated image.

    Returns:
        3-channel BGR NumPy array of shape (height, width, 3) and dtype uint8.
    """
    image = np.full((height, width, 3), 128, dtype=np.uint8)

    # Bright white rectangle in the top-left quadrant
    cv2.rectangle(image, (50, 40), (180, 140), (255, 255, 255), thickness=-1)

    # Dark rectangle in the bottom-right quadrant
    cv2.rectangle(image, (220, 160), (350, 260), (10, 10, 10), thickness=-1)

    return image


@pytest.fixture
def synthetic_bgr_image() -> np.ndarray:
    """Fixture providing a deterministic 300x400 BGR synthetic image."""
    return create_synthetic_test_image(300, 400)


@pytest.fixture
def synthetic_gray_image(synthetic_bgr_image: np.ndarray) -> np.ndarray:
    """Fixture providing a grayscale version of the synthetic image."""
    return to_grayscale(synthetic_bgr_image)


# ============================================================================
# Preprocessing Tests
# ============================================================================


def test_to_grayscale_valid(synthetic_bgr_image: np.ndarray):
    """Test that a valid 3-channel BGR image is converted to a 2D grayscale image."""
    gray = to_grayscale(synthetic_bgr_image)

    assert isinstance(gray, np.ndarray)
    assert gray.ndim == 2
    assert gray.shape == (300, 400)
    assert gray.dtype == np.uint8

    # Verify background and rectangle gray levels
    assert gray[10, 10] == 128
    assert gray[90, 100] == 255
    assert gray[200, 300] == 10


def test_to_grayscale_invalid_inputs():
    """Test that invalid inputs to to_grayscale raise PreprocessingError."""
    with pytest.raises(PreprocessingError, match="must be a valid NumPy array"):
        to_grayscale(None)  # type: ignore

    with pytest.raises(PreprocessingError, match="must be a valid NumPy array"):
        to_grayscale([1, 2, 3])  # type: ignore

    with pytest.raises(PreprocessingError, match="array is empty"):
        to_grayscale(np.array([], dtype=np.uint8))

    # Single-channel 2D image
    with pytest.raises(PreprocessingError, match="Expected a 3-channel BGR image"):
        to_grayscale(np.zeros((100, 100), dtype=np.uint8))

    # 4-channel BGRA image
    with pytest.raises(PreprocessingError, match="Expected a 3-channel BGR image"):
        to_grayscale(np.zeros((100, 100, 4), dtype=np.uint8))

    # Float dtype array
    with pytest.raises(PreprocessingError, match="Expected image array dtype uint8"):
        to_grayscale(np.zeros((100, 100, 3), dtype=np.float32))


def test_apply_gaussian_blur_valid(synthetic_gray_image: np.ndarray):
    """Test Gaussian blur with various positive odd kernels."""
    for ksize in [1, 3, 5, 9]:
        blurred = apply_gaussian_blur(synthetic_gray_image, kernel_size=ksize)
        assert isinstance(blurred, np.ndarray)
        assert blurred.shape == synthetic_gray_image.shape
        assert blurred.dtype == np.uint8

    # Also works on 3-channel BGR image
    bgr = create_synthetic_test_image(100, 100)
    blurred_bgr = apply_gaussian_blur(bgr, kernel_size=5)
    assert blurred_bgr.shape == (100, 100, 3)
    assert blurred_bgr.dtype == np.uint8


def test_apply_gaussian_blur_invalid_kernels(synthetic_gray_image: np.ndarray):
    """Test rejection of non-integer, non-positive, and even kernel sizes."""
    # Even kernels
    for even_k in [0, 2, 4, 6]:
        with pytest.raises(PreprocessingError):
            apply_gaussian_blur(synthetic_gray_image, kernel_size=even_k)

    # Negative kernels
    for neg_k in [-1, -3, -5]:
        with pytest.raises(PreprocessingError, match="must be positive"):
            apply_gaussian_blur(synthetic_gray_image, kernel_size=neg_k)

    # Non-integer types
    with pytest.raises(PreprocessingError, match="must be an integer"):
        apply_gaussian_blur(synthetic_gray_image, kernel_size=3.5)  # type: ignore

    with pytest.raises(PreprocessingError, match="must be an integer"):
        apply_gaussian_blur(synthetic_gray_image, kernel_size=True)  # type: ignore

    with pytest.raises(PreprocessingError, match="must be an integer"):
        apply_gaussian_blur(synthetic_gray_image, kernel_size="5")  # type: ignore


def test_apply_gaussian_blur_invalid_image():
    """Test that invalid images raise PreprocessingError."""
    with pytest.raises(PreprocessingError, match="must be a valid NumPy array"):
        apply_gaussian_blur(None, kernel_size=5)  # type: ignore

    with pytest.raises(PreprocessingError, match="array is empty"):
        apply_gaussian_blur(np.array([]), kernel_size=5)


def test_preprocess_image_pipeline(synthetic_bgr_image: np.ndarray):
    """Test the complete preprocessing pipeline (grayscale + blur)."""
    preprocessed = preprocess_image(synthetic_bgr_image, blur_kernel=5)

    assert isinstance(preprocessed, np.ndarray)
    assert preprocessed.ndim == 2
    assert preprocessed.shape == (300, 400)
    assert preprocessed.dtype == np.uint8

    # Invalid kernel raises clear error
    with pytest.raises(PreprocessingError, match="must be an odd number"):
        preprocess_image(synthetic_bgr_image, blur_kernel=4)


# ============================================================================
# Edge Detection Tests
# ============================================================================


def test_detect_edges_valid(synthetic_gray_image: np.ndarray):
    """Test Canny edge detection produces valid binary edges with expected dimensions."""
    edges = detect_edges(synthetic_gray_image, threshold1=50, threshold2=150)

    assert isinstance(edges, np.ndarray)
    assert edges.ndim == 2
    assert edges.shape == synthetic_gray_image.shape
    assert edges.dtype == np.uint8

    # Edge image values should be binary (0 or 255)
    unique_vals = set(np.unique(edges))
    assert unique_vals.issubset({0, 255})

    # High-contrast rectangles must produce edges
    assert np.count_nonzero(edges) > 0


def test_detect_edges_invalid_inputs():
    """Test that invalid image inputs raise EdgeDetectionError."""
    with pytest.raises(EdgeDetectionError, match="must be a valid NumPy array"):
        detect_edges(None)  # type: ignore

    with pytest.raises(EdgeDetectionError, match="must be a valid NumPy array"):
        detect_edges("invalid")  # type: ignore

    with pytest.raises(EdgeDetectionError, match="array is empty"):
        detect_edges(np.array([], dtype=np.uint8))

    # 3-channel color image
    with pytest.raises(EdgeDetectionError, match="Expected a 2D single-channel grayscale image"):
        detect_edges(np.zeros((100, 100, 3), dtype=np.uint8))

    # Float dtype array
    with pytest.raises(EdgeDetectionError, match="Expected grayscale image array dtype uint8"):
        detect_edges(np.zeros((100, 100), dtype=np.float32))


def test_detect_edges_threshold_validation(synthetic_gray_image: np.ndarray):
    """Test validation of Canny lower and upper thresholds."""
    # Lower threshold >= upper threshold
    with pytest.raises(EdgeDetectionError, match="must be strictly less than upper threshold"):
        detect_edges(synthetic_gray_image, threshold1=150, threshold2=50)

    with pytest.raises(EdgeDetectionError, match="must be strictly less than upper threshold"):
        detect_edges(synthetic_gray_image, threshold1=100, threshold2=100)

    # Negative lower threshold
    with pytest.raises(EdgeDetectionError, match="must be >= 0"):
        detect_edges(synthetic_gray_image, threshold1=-10, threshold2=100)

    # Non-numeric or boolean threshold
    with pytest.raises(EdgeDetectionError, match="must be a number"):
        detect_edges(synthetic_gray_image, threshold1=True, threshold2=100)  # type: ignore

    with pytest.raises(EdgeDetectionError, match="must be a number"):
        detect_edges(synthetic_gray_image, threshold1=50, threshold2="200")  # type: ignore


def test_apply_morphological_closing():
    """Test morphological closing bridges small gaps in edge maps."""
    # Create an edge line with a 1-pixel gap
    edge_map = np.zeros((20, 20), dtype=np.uint8)
    edge_map[10, 0:8] = 255
    # gap at (10, 8)
    edge_map[10, 9:18] = 255

    assert edge_map[10, 8] == 0

    closed = apply_morphological_closing(edge_map, kernel_size=3)
    assert isinstance(closed, np.ndarray)
    assert closed.shape == edge_map.shape
    # Gap should now be filled
    assert closed[10, 8] == 255


def test_apply_morphological_closing_validation():
    """Test morphological closing parameter and input validation."""
    edge_map = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(EdgeDetectionError, match="must be a valid NumPy array"):
        apply_morphological_closing(None)  # type: ignore

    with pytest.raises(EdgeDetectionError, match="array is empty"):
        apply_morphological_closing(np.array([], dtype=np.uint8))

    with pytest.raises(EdgeDetectionError, match="Expected a 2D single-channel edge image"):
        apply_morphological_closing(np.zeros((20, 20, 3), dtype=np.uint8))

    # Even kernel
    with pytest.raises(EdgeDetectionError, match="must be an odd number"):
        apply_morphological_closing(edge_map, kernel_size=4)

    # Negative / zero kernel
    with pytest.raises(EdgeDetectionError, match="must be positive"):
        apply_morphological_closing(edge_map, kernel_size=0)

    with pytest.raises(EdgeDetectionError, match="must be positive"):
        apply_morphological_closing(edge_map, kernel_size=-3)

    # Non-integer type
    with pytest.raises(EdgeDetectionError, match="must be an integer"):
        apply_morphological_closing(edge_map, kernel_size=True)  # type: ignore


def test_detect_edges_with_closing(synthetic_gray_image: np.ndarray):
    """Test detect_edges with optional morphological closing enabled and adjusted."""
    edges_no_closing = detect_edges(
        synthetic_gray_image,
        threshold1=50,
        threshold2=150,
        apply_closing=False,
    )
    edges_with_closing = detect_edges(
        synthetic_gray_image,
        threshold1=50,
        threshold2=150,
        apply_closing=True,
        closing_kernel_size=3,
    )

    assert edges_with_closing.shape == edges_no_closing.shape
    assert edges_with_closing.dtype == np.uint8

    # Invalid closing kernel should be rejected
    with pytest.raises(EdgeDetectionError, match="must be an odd number"):
        detect_edges(
            synthetic_gray_image,
            threshold1=50,
            threshold2=150,
            apply_closing=True,
            closing_kernel_size=4,
        )


def test_full_pipeline_on_sample_disk_image():
    """Test loading, preprocessing, and edge detecting the repository's sample image."""
    sample_path = Path("data/sample_images/sample_objects.png")
    if not sample_path.exists():
        pytest.skip("data/sample_images/sample_objects.png not found")

    image = load_image(sample_path)
    assert image.ndim == 3

    preprocessed = preprocess_image(image, blur_kernel=5)
    assert preprocessed.shape == image.shape[:2]

    edges = detect_edges(preprocessed, threshold1=50, threshold2=150, apply_closing=True)
    assert edges.shape == image.shape[:2]
    assert np.count_nonzero(edges) > 0

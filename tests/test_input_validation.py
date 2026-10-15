"""Tests for app.image_input module."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.image_input import (
    CameraError,
    ImageNotFoundError,
    ImageReadError,
    UnsupportedFormatError,
    capture_frame,
    load_image,
    validate_image_array,
)


@pytest.fixture
def sample_images(tmp_path: Path):
    """Generate temporary test images in PNG, JPG, and JPEG formats."""
    # Create a 100x150 3-channel test image (solid color)
    img_data = np.zeros((100, 150, 3), dtype=np.uint8)
    img_data[:, :] = (128, 64, 255)  # BGR

    png_path = tmp_path / "test.png"
    jpg_path = tmp_path / "test.jpg"
    jpeg_path = tmp_path / "test.jpeg"

    cv2.imwrite(str(png_path), img_data)
    cv2.imwrite(str(jpg_path), img_data)
    cv2.imwrite(str(jpeg_path), img_data)

    return {
        "png": png_path,
        "jpg": jpg_path,
        "jpeg": jpeg_path,
        "shape": (100, 150, 3),
    }


def test_load_valid_images(sample_images):
    """Test loading valid PNG, JPG, and JPEG images."""
    for fmt in ["png", "jpg", "jpeg"]:
        img = load_image(sample_images[fmt])
        assert isinstance(img, np.ndarray)
        assert img.shape == sample_images["shape"]
        assert img.dtype == np.uint8


def test_load_nonexistent_file(tmp_path: Path):
    """Test that a non-existent file raises ImageNotFoundError."""
    non_existent = tmp_path / "does_not_exist.png"
    with pytest.raises(ImageNotFoundError, match="does not exist"):
        load_image(non_existent)


def test_load_empty_path():
    """Test that an empty path raises ImageNotFoundError."""
    with pytest.raises(ImageNotFoundError, match="must not be empty"):
        load_image("")


def test_load_unsupported_extension(tmp_path: Path):
    """Test that an unsupported extension raises UnsupportedFormatError."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("not an image")
    with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
        load_image(txt_file)

    bmp_file = tmp_path / "sample.bmp"
    bmp_file.write_bytes(b"BMfakebmpheader")
    with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
        load_image(bmp_file)


def test_load_corrupt_file(tmp_path: Path):
    """Test that an empty or corrupted image file raises ImageReadError."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"corrupted binary data")
    with pytest.raises(ImageReadError, match="Failed to decode"):
        load_image(corrupt_file)


def test_validate_image_array():
    """Test array validation helper with various invalid shapes and types."""
    with pytest.raises(ImageReadError, match="not a valid NumPy array"):
        validate_image_array(None)  # type: ignore

    with pytest.raises(ImageReadError, match="not a valid NumPy array"):
        validate_image_array([1, 2, 3])  # type: ignore

    with pytest.raises(ImageReadError, match="array is empty"):
        validate_image_array(np.array([], dtype=np.uint8))

    # Grayscale 2D array
    with pytest.raises(ImageReadError, match="expected 3-channel BGR"):
        validate_image_array(np.zeros((100, 100), dtype=np.uint8))

    # 4-channel BGRA array
    with pytest.raises(ImageReadError, match="expected 3-channel BGR"):
        validate_image_array(np.zeros((100, 100, 4), dtype=np.uint8))


def test_capture_frame_success():
    """Test successful webcam capture using a mock VideoCapture."""
    fake_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, fake_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        frame = capture_frame(camera_index=0)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        mock_cap.release.assert_called_once()


def test_capture_frame_camera_cannot_open():
    """Test that a failure to open the camera raises CameraError and releases the resource."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(CameraError, match="Cannot open camera index"):
            capture_frame(camera_index=0)
        mock_cap.release.assert_called_once()


def test_capture_frame_read_failure():
    """Test that a failure during frame read raises CameraError and releases the resource."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(CameraError, match="Failed to capture a valid frame"):
            capture_frame(camera_index=0)
        mock_cap.release.assert_called_once()

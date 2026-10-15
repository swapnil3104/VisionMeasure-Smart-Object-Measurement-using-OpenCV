"""Tests for the YOLO-based detection and measurement engine."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.yolo_detector import (
    YOLODetection,
    YOLODetectionError,
    YOLOMeasurementEngine,
)


@pytest.fixture(scope="module")
def yolo_engine():
    """Module-level fixture to instantiate YOLO measurement engine once."""
    return YOLOMeasurementEngine("yolov8n.pt")


@pytest.fixture
def synthetic_scene():
    """Create a synthetic test image with a high-contrast white rectangle on dark background."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a prominent white rectangle (simulating a card or box)
    cv2.rectangle(img, (150, 120), (350, 240), (255, 255, 255), -1)
    return img


def test_yolo_engine_initialization(yolo_engine):
    """Test that the YOLO engine initializes and exposes its model."""
    assert yolo_engine is not None
    assert yolo_engine.model is not None
    assert hasattr(yolo_engine.model, "names")


def test_yolo_detection_empty_input(yolo_engine):
    """Test that passing None or an empty array raises YOLODetectionError."""
    with pytest.raises(YOLODetectionError):
        yolo_engine.detect_and_measure(None)

    with pytest.raises(YOLODetectionError):
        yolo_engine.detect_and_measure(np.array([], dtype=np.uint8))


def test_yolo_detection_structure(yolo_engine, synthetic_scene):
    """Test the structure of detection output."""
    detections = yolo_engine.detect_and_measure(
        synthetic_scene,
        conf_threshold=0.1,
        mode="standard",
    )
    # The return value should always be a list
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, YOLODetection)
        assert det.length_px >= det.width_px
        assert det.area_px2 >= 0
        assert len(det.bbox) == 4
        assert det.box_points is not None


def test_yolo_calibration_calculation(yolo_engine):
    """Test physical dimension calculations when a calibration scale is provided."""
    # Create an artificial detection record
    det = YOLODetection(
        object_id=1,
        class_id=0,
        class_name="cell phone",
        confidence=0.92,
        bbox=[100, 100, 300, 200],
        box_points=np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.float32),
        center=(200.0, 150.0),
        length_px=200.0,
        width_px=100.0,
        area_px2=20000.0,
        angle_deg=0.0,
        is_calibrated=True,
        length_cm=200.0 / 40.0,  # 5.0 cm
        width_cm=100.0 / 40.0,   # 2.5 cm
        length_mm=50.0,
        width_mm=25.0,
        length_in=5.0 / 2.54,
        width_in=2.5 / 2.54,
        area_cm2=12.5,
        pixels_per_cm=40.0,
        is_oriented=False,
    )

    assert det.is_calibrated is True
    assert det.length_cm == 5.0
    assert det.width_cm == 2.5
    assert det.length_mm == 50.0
    assert det.width_mm == 25.0
    assert pytest.approx(det.area_cm2) == 12.5

    # Check dimension formatting
    assert "5.00 x 2.50 cm" in det.format_dimension_string("cm")
    assert "50.0 x 25.0 mm" in det.format_dimension_string("mm")
    assert "in" in det.format_dimension_string("in")

    # Serialization
    d_dict = det.to_dict()
    assert d_dict["object_id"] == 1
    assert d_dict["class_name"] == "cell phone"
    assert d_dict["length_cm"] == 5.0


def test_yolo_annotation_preserves_original(yolo_engine, synthetic_scene):
    """Test that annotate_detections returns a new annotated copy and does not mutate input."""
    original_copy = synthetic_scene.copy()
    dummy_det = YOLODetection(
        object_id=1,
        class_id=39,
        class_name="bottle",
        confidence=0.88,
        bbox=[50, 50, 150, 250],
        box_points=np.array([[50, 50], [150, 50], [150, 250], [50, 250]], dtype=np.float32),
        center=(100.0, 150.0),
        length_px=200.0,
        width_px=100.0,
        area_px2=20000.0,
        angle_deg=0.0,
        is_calibrated=True,
        length_cm=10.0,
        width_cm=5.0,
        pixels_per_cm=20.0,
    )

    annotated = yolo_engine.annotate_detections(
        synthetic_scene,
        [dummy_det],
        unit="cm",
        show_dimensions=True,
        show_labels=True,
    )

    # Input image was not mutated
    assert np.array_equal(synthetic_scene, original_copy)
    # Annotated image has same shape but modified pixels
    assert annotated.shape == synthetic_scene.shape
    assert not np.array_equal(annotated, synthetic_scene)

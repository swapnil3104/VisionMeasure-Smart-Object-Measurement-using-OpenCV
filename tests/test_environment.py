"""Initial environment and sanity checks."""
import cv2
import numpy as np
from models.measurement_result import MeasurementResult


def test_dependencies_import():
    """Verify core dependencies are functional."""
    assert cv2.__version__ is not None
    assert np.__version__ is not None


def test_measurement_result_model():
    """Verify MeasurementResult dataclass structure and property calculation."""
    res = MeasurementResult(
        object_id=1,
        length_px=200.0,
        width_px=100.0,
        contour_area_px2=20000.0,
        length_cm=10.0,
        width_cm=5.0,
        pixels_per_cm=20.0,
    )
    assert res.object_id == 1
    assert res.length_px == 200.0
    assert res.width_px == 100.0
    assert res.length_cm == 10.0
    assert res.width_cm == 5.0
    assert res.length_mm == 100.0
    assert res.width_mm == 50.0

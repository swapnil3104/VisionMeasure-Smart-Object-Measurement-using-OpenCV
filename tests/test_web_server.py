"""Tests for the VisionMeasure REST API and web application server."""
from __future__ import annotations

import base64

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.web_server import app


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient for FastAPI."""
    return TestClient(app)


@pytest.fixture
def test_image_b64():
    """Generate a test image encoded as base64."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def test_health_check(client):
    """Test /api/health endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model" in data


def test_index_page_serves_html(client):
    """Test that GET / returns the HTML user interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "VisionMeasure AI" in response.text
    assert "mainImageCanvas" in response.text


def test_sample_images_endpoint(client):
    """Test /api/sample_images returns list of samples."""
    response = client.get("/api/sample_images")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert isinstance(data["samples"], list)


def test_calibrate_points_endpoint(client):
    """Test /api/calibrate/points calculates scale accurately."""
    payload = {
        "x1": 100.0,
        "y1": 100.0,
        "x2": 300.0,
        "y2": 100.0,  # 200 px horizontal span
        "known_length": 5.0,  # 5.0 cm -> 40 px/cm
        "unit": "cm",
    }
    response = client.post("/api/calibrate/points", data=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["pixel_distance"] == 200.0
    assert data["pixels_per_cm"] == 40.0
    assert data["pixels_per_mm"] == 4.0


def test_calibrate_object_endpoint(client):
    """Test /api/calibrate/object calculates scale factor from object px."""
    payload = {
        "object_pixel_length": 342.4,
        "known_length": 8.56,  # Standard credit card length cm
        "unit": "cm",
    }
    response = client.post("/api/calibrate/object", data=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert pytest.approx(data["pixels_per_cm"], rel=1e-3) == 40.0


def test_detect_endpoint_with_base64(client, test_image_b64):
    """Test /api/detect with base64 image data."""
    payload = {
        "image_base64": test_image_b64,
        "conf_threshold": 0.25,
        "mode": "standard",
        "pixels_per_cm": 40.0,
        "unit": "cm",
    }
    response = client.post("/api/detect", data=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "detections" in data
    assert "annotated_image" in data
    assert data["image_width"] == 300
    assert data["image_height"] == 200
    assert data["is_calibrated"] is True


def test_detect_endpoint_contour_mode(client, test_image_b64):
    """Test /api/detect in OpenCV contour mode."""
    payload = {
        "image_base64": test_image_b64,
        "mode": "contour",
        "pixels_per_cm": 40.0,
        "unit": "cm",
        "min_area_px": 50.0,
    }
    response = client.post("/api/detect", data=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["detection_mode"] == "OpenCV Contour"
    assert data["total_detected"] >= 1


def test_export_csv_endpoint(client):
    """Test /api/export/csv generates a valid CSV file."""
    sample_det = {
        "object_id": 1,
        "class_name": "bottle",
        "confidence": 0.95,
        "length_px": 200.0,
        "width_px": 80.0,
        "area_px2": 16000.0,
        "angle_deg": 0.0,
        "is_calibrated": True,
        "length_cm": 5.0,
        "width_cm": 2.0,
        "length_mm": 50.0,
        "width_mm": 20.0,
        "length_in": 1.97,
        "width_in": 0.79,
        "area_cm2": 10.0,
        "pixels_per_cm": 40.0,
        "bbox": [10, 20, 90, 220],
    }
    response = client.post("/api/export/csv", json={"detections": [sample_det]})
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "Object ID,Class Name" in response.text
    assert "bottle" in response.text


def test_export_json_endpoint(client):
    """Test /api/export/json returns downloadable JSON file."""
    response = client.post("/api/export/json", json={"detections": [{"id": 1}]})
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

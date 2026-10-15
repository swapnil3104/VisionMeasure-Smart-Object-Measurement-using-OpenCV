"""Web Server application for VisionMeasure.

Provides a fast, modern REST API and web application replacing the legacy desktop GUI.
Powered by FastAPI, Uvicorn, Ultralytics YOLO, and OpenCV.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np

from app.contour_detection import detect_contours
from app.edge_detection import detect_edges
from app.preprocessing import preprocess_image
from app.yolo_detector import YOLOMeasurementEngine

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
DATA_DIR = BASE_DIR / "data"

# Create directories if needed
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="VisionMeasure AI — Smart Object Dimension Measurement",
    description="Web application for AI-powered real-time object detection with bounding boxes and calibrated physical measurements.",
    version="2.0.0",
)

# Enable CORS for maximum flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global YOLO engine cache (keyed by model name)
_yolo_engines: Dict[str, YOLOMeasurementEngine] = {}
_current_model_name: str = "yolov8m.pt"


def get_yolo_engine(model_name: Optional[str] = None) -> YOLOMeasurementEngine:
    """Retrieve or lazily initialize the YOLO measurement engine for a specific model."""
    global _yolo_engines, _current_model_name
    target = (model_name or _current_model_name).strip()
    if target not in _yolo_engines:
        _yolo_engines[target] = YOLOMeasurementEngine(target)
    _current_model_name = target
    return _yolo_engines[target]


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file or format.")
    return img


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode a base64-encoded image string (with or without data URI scheme) into OpenCV BGR."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        return decode_image_bytes(img_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {exc}")


def encode_image_base64(img_bgr: np.ndarray, format: str = ".jpg", quality: int = 90) -> str:
    """Encode an OpenCV BGR image into a base64 data URI string."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if format in (".jpg", ".jpeg") else []
    success, buffer = cv2.imencode(format, img_bgr, params)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image to base64.")
    b64 = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


# -----------------------------------------------------------------------------
# Web Page Route
# -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Render the main VisionMeasure web application UI."""
    return templates.TemplateResponse(request, "index.html")


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    """System health check and version verification."""
    return {
        "status": "healthy",
        "app": "VisionMeasure AI",
        "version": "2.0.0",
        "model": get_yolo_engine().model_name,
    }


@app.get("/api/models")
async def list_available_models():
    """Return available YOLO models with accuracy and speed specifications."""
    return {
        "active_model": _current_model_name,
        "models": [
            {
                "id": "yolov8m.pt",
                "name": "YOLOv8 Medium",
                "badge": "High Accuracy (Recommended)",
                "accuracy": "50.2 mAP",
                "speed": "Balanced",
                "params": "25.9M",
                "description": "Recommended default. High accuracy and recall for reliable object detection and dimensioning.",
            },
            {
                "id": "yolov8x.pt",
                "name": "YOLOv8 Extra-Large",
                "badge": "Maximum Precision",
                "accuracy": "53.9 mAP",
                "speed": "Compute-Intensive",
                "params": "68.2M",
                "description": "Highest accuracy among YOLO models. Pinpoint bounding box localization and detection of challenging objects.",
            },
            {
                "id": "yolov8s.pt",
                "name": "YOLOv8 Small",
                "badge": "Fast & Balanced",
                "accuracy": "44.9 mAP",
                "speed": "Fast",
                "params": "11.2M",
                "description": "Balanced speed and accuracy, very lightweight on CPU.",
            },
            {
                "id": "yolov8n.pt",
                "name": "YOLOv8 Nano",
                "badge": "Lightweight",
                "accuracy": "37.3 mAP",
                "speed": "Ultra-Fast",
                "params": "3.2M",
                "description": "Lowest resource usage for edge devices.",
            },
        ],
    }


@app.get("/api/classes")
async def get_detectable_classes():
    """Return all COCO detectable classes supported by YOLO."""
    engine = get_yolo_engine()
    names = engine.model.names if engine.model and hasattr(engine.model, "names") else {}
    return {"classes": list(names.values()) if isinstance(names, dict) else names}


@app.get("/api/sample_images")
async def list_sample_images():
    """List preloaded sample images for instant one-click demonstration."""
    sample_dir = DATA_DIR / "sample_images"
    samples = []
    seen_stems = set()
    if sample_dir.is_dir():
        for f in sorted(sample_dir.glob("*.*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.stem not in seen_stems:
                seen_stems.add(f.stem)
                samples.append({
                    "id": f.stem,
                    "filename": f.name,
                    "name": f.stem.replace("_", " ").title(),
                    "url": f"/api/sample_images/{f.name}",
                })
    return {"samples": samples}


@app.get("/api/sample_images/{filename}")
async def get_sample_image(filename: str):
    """Serve a sample image file."""
    file_path = DATA_DIR / "sample_images" / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    media_type = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


@app.post("/api/calibrate/points")
async def calibrate_by_two_points(
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    known_length: float = Form(...),
    unit: str = Form("cm"),
):
    """Calculate scale factor (pixels_per_cm) from two user-selected pixel points."""
    if known_length <= 0:
        raise HTTPException(status_code=400, detail="Known real-world length must be greater than zero.")

    pixel_distance = math.hypot(x2 - x1, y2 - y1)
    if pixel_distance < 1.0:
        raise HTTPException(status_code=400, detail="Selected points are too close together.")

    u = unit.lower().strip()
    if u == "mm":
        real_cm = known_length / 10.0
    elif u in ("in", "inch", "inches"):
        real_cm = known_length * 2.54
    else:
        real_cm = known_length

    pixels_per_cm = pixel_distance / real_cm
    pixels_per_mm = pixels_per_cm / 10.0
    pixels_per_in = pixels_per_cm * 2.54

    return {
        "success": True,
        "pixel_distance": round(pixel_distance, 2),
        "known_length": known_length,
        "unit": unit,
        "pixels_per_cm": round(pixels_per_cm, 4),
        "pixels_per_mm": round(pixels_per_mm, 4),
        "pixels_per_in": round(pixels_per_in, 4),
        "status_message": f"Calibrated: {pixels_per_cm:.2f} px/cm ({pixel_distance:.1f} px = {known_length} {unit})",
    }


@app.post("/api/calibrate/object")
async def calibrate_by_object(
    object_pixel_length: float = Form(...),
    known_length: float = Form(...),
    unit: str = Form("cm"),
):
    """Calculate scale factor (pixels_per_cm) from a detected object's pixel dimension."""
    if known_length <= 0 or object_pixel_length <= 0:
        raise HTTPException(status_code=400, detail="Dimensions must be strictly positive.")

    u = unit.lower().strip()
    if u == "mm":
        real_cm = known_length / 10.0
    elif u in ("in", "inch", "inches"):
        real_cm = known_length * 2.54
    else:
        real_cm = known_length

    pixels_per_cm = object_pixel_length / real_cm
    return {
        "success": True,
        "pixels_per_cm": round(pixels_per_cm, 4),
        "pixels_per_mm": round(pixels_per_cm / 10.0, 4),
        "pixels_per_in": round(pixels_per_cm * 2.54, 4),
        "status_message": f"Calibrated: {pixels_per_cm:.2f} px/cm ({object_pixel_length:.1f} px = {known_length} {unit})",
    }


@app.post("/api/detect")
async def detect_and_measure_objects(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    model_name: str = Form("yolov8m.pt"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    mode: str = Form("oriented"),  # 'oriented', 'standard', or 'contour'
    pixels_per_cm: Optional[float] = Form(None),
    unit: str = Form("cm"),
    target_classes: Optional[str] = Form(None),
    show_dimension_lines: bool = Form(True),
    palette_mode: str = Form("dynamic"),
    min_area_px: float = Form(100.0),
    canny_low: int = Form(50),
    canny_high: int = Form(150),
):
    """Run object detection with bounding boxes and calculate dimensions."""
    # Obtain input image
    if file is not None and file.filename:
        content = await file.read()
        img_bgr = decode_image_bytes(content)
    elif image_base64:
        img_bgr = decode_base64_image(image_base64)
    else:
        raise HTTPException(status_code=400, detail="No image provided. Upload a file or provide base64 data.")

    h, w = img_bgr.shape[:2]
    classes_filter = [c.strip() for c in target_classes.split(",")] if target_classes else None

    # Handle Contour (traditional CV) mode if requested
    if mode == "contour":
        return _run_contour_detection_pipeline(
            img_bgr, pixels_per_cm, unit, min_area_px, canny_low, canny_high
        )

    # YOLO AI Object Detection & Measurement pipeline
    engine = get_yolo_engine(model_name)
    detections = engine.detect_and_measure(
        image_bgr=img_bgr,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        pixels_per_cm=pixels_per_cm,
        mode=mode,
        target_classes=classes_filter,
    )

    # Render annotated image
    annotated_bgr = engine.annotate_detections(
        image_bgr=img_bgr,
        detections=detections,
        unit=unit,
        show_dimensions=True,
        show_labels=True,
        show_dimension_lines=show_dimension_lines,
        palette_mode=palette_mode,
    )

    annotated_b64 = encode_image_base64(annotated_bgr, format=".jpg", quality=88)

    # Summary metrics
    total_count = len(detections)
    largest = max(detections, key=lambda d: d.area_px2).to_dict() if detections else None
    smallest = min(detections, key=lambda d: d.area_px2).to_dict() if detections else None
    avg_len_px = sum(d.length_px for d in detections) / total_count if total_count > 0 else 0.0

    return {
        "success": True,
        "image_width": w,
        "image_height": h,
        "detection_mode": f"YOLO ({mode})",
        "model_name": engine.model_name,
        "total_detected": total_count,
        "is_calibrated": bool(pixels_per_cm and pixels_per_cm > 0),
        "pixels_per_cm": pixels_per_cm,
        "unit": unit,
        "detections": [d.to_dict() for d in detections],
        "annotated_image": annotated_b64,
        "summary": {
            "total_objects": total_count,
            "largest_object": largest,
            "smallest_object": smallest,
            "avg_length_px": round(avg_len_px, 2),
            "avg_length_cm": round(avg_len_px / pixels_per_cm, 2) if pixels_per_cm else None,
        },
    }


def _run_contour_detection_pipeline(
    img_bgr: np.ndarray,
    pixels_per_cm: Optional[float],
    unit: str,
    min_area_px: float,
    canny_low: int,
    canny_high: int,
) -> Dict[str, Any]:
    """Execute traditional OpenCV Canny and contour detection pipeline."""
    h, w = img_bgr.shape[:2]
    pre = preprocess_image(img_bgr)
    edges = detect_edges(pre, threshold1=canny_low, threshold2=canny_high)
    geometries = detect_contours(edges, min_area=min_area_px)

    detections = []
    annotated = img_bgr.copy()

    for idx, geo in enumerate(geometries, start=1):
        len_px = geo.length_px
        wid_px = geo.width_px
        area_px2 = geo.contour_area_px2

        len_cm = (len_px / pixels_per_cm) if pixels_per_cm else None
        wid_cm = (wid_px / pixels_per_cm) if pixels_per_cm else None
        len_mm = (len_cm * 10.0) if len_cm else None
        wid_mm = (wid_cm * 10.0) if wid_cm else None
        len_in = (len_cm / 2.54) if len_cm else None
        wid_in = (wid_cm / 2.54) if wid_cm else None
        area_cm2 = (len_cm * wid_cm) if (len_cm and wid_cm) else None

        det_dict = {
            "object_id": idx,
            "class_id": 0,
            "class_name": "contour_object",
            "confidence": 1.0,
            "bbox": [
                int(round(geo.center[0] - wid_px / 2)),
                int(round(geo.center[1] - len_px / 2)),
                int(round(geo.center[0] + wid_px / 2)),
                int(round(geo.center[1] + len_px / 2)),
            ],
            "box_points": geo.box_points.tolist() if geo.box_points is not None else [],
            "center": [round(float(geo.center[0]), 2), round(float(geo.center[1]), 2)],
            "length_px": round(len_px, 2),
            "width_px": round(wid_px, 2),
            "area_px2": round(area_px2, 2),
            "angle_deg": round(geo.angle_deg, 1),
            "is_calibrated": bool(pixels_per_cm and pixels_per_cm > 0),
            "length_cm": round(len_cm, 3) if len_cm else None,
            "width_cm": round(wid_cm, 3) if wid_cm else None,
            "length_mm": round(len_mm, 2) if len_mm else None,
            "width_mm": round(wid_mm, 2) if wid_mm else None,
            "length_in": round(len_in, 3) if len_in else None,
            "width_in": round(wid_in, 3) if wid_in else None,
            "area_cm2": round(area_cm2, 3) if area_cm2 else None,
            "pixels_per_cm": pixels_per_cm,
            "is_oriented": True,
        }
        detections.append(det_dict)

        # Draw contour minAreaRect
        pts = np.int32(geo.box_points)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 128), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(annotated, (int(geo.center[0]), int(geo.center[1])), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Label
        label = f"#{idx}: "
        if pixels_per_cm:
            label += f"{len_cm:.2f} x {wid_cm:.2f} {unit}"
        else:
            label += f"{len_px:.1f} x {wid_px:.1f} px"

        cv2.putText(
            annotated,
            label,
            (int(geo.center[0] - 30), max(20, int(geo.center[1] - 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 128),
            1,
            lineType=cv2.LINE_AA,
        )

    annotated_b64 = encode_image_base64(annotated, format=".jpg", quality=88)

    return {
        "success": True,
        "image_width": w,
        "image_height": h,
        "detection_mode": "OpenCV Contour",
        "total_detected": len(detections),
        "is_calibrated": bool(pixels_per_cm and pixels_per_cm > 0),
        "pixels_per_cm": pixels_per_cm,
        "unit": unit,
        "detections": detections,
        "annotated_image": annotated_b64,
        "summary": {
            "total_objects": len(detections),
            "largest_object": max(detections, key=lambda d: d["area_px2"]) if detections else None,
            "smallest_object": min(detections, key=lambda d: d["area_px2"]) if detections else None,
            "avg_length_px": round(sum(d["length_px"] for d in detections) / len(detections), 2) if detections else 0.0,
            "avg_length_cm": round(sum(d["length_px"] for d in detections) / (len(detections) * pixels_per_cm), 2) if (detections and pixels_per_cm) else None,
        },
    }


@app.post("/api/export/csv")
async def export_measurements_csv(payload: Dict[str, Any]):
    """Export current measurement results to a formatted CSV file."""
    detections = payload.get("detections", [])
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Object ID",
        "Class Name",
        "Confidence",
        "Length (px)",
        "Width (px)",
        "Area (px²)",
        "Rotation (°)",
        "Is Calibrated",
        "Length (cm)",
        "Width (cm)",
        "Length (mm)",
        "Width (mm)",
        "Length (in)",
        "Width (in)",
        "Area (cm²)",
        "Scale (px/cm)",
        "Bounding Box [x1, y1, x2, y2]",
    ])

    for d in detections:
        writer.writerow([
            d.get("object_id", ""),
            d.get("class_name", ""),
            d.get("confidence", ""),
            d.get("length_px", ""),
            d.get("width_px", ""),
            d.get("area_px2", ""),
            d.get("angle_deg", ""),
            d.get("is_calibrated", False),
            d.get("length_cm", ""),
            d.get("width_cm", ""),
            d.get("length_mm", ""),
            d.get("width_mm", ""),
            d.get("length_in", ""),
            d.get("width_in", ""),
            d.get("area_cm2", ""),
            d.get("pixels_per_cm", ""),
            json.dumps(d.get("bbox", [])),
        ])

    output.seek(0)
    response = Response(content=output.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=visionmeasure_results.csv"
    return response


@app.post("/api/export/json")
async def export_measurements_json(payload: Dict[str, Any]):
    """Export measurements as structured JSON file."""
    output = json.dumps(payload, indent=2)
    response = Response(content=output, media_type="application/json")
    response.headers["Content-Disposition"] = "attachment; filename=visionmeasure_results.json"
    return response


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Launch the VisionMeasure web application server with uvicorn."""
    import uvicorn

    print(f"\n==================================================================")
    print(f" VisionMeasure AI — Web Application")
    print(f" URL: http://{host}:{port}")
    print(f" Swagger API Docs: http://{host}:{port}/docs")
    print(f"==================================================================\n")
    uvicorn.run("app.web_server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()

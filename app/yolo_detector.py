"""YOLO-based Object Detection and Dimension Measurement Engine for VisionMeasure.

Combines state-of-the-art deep learning object detection (Ultralytics YOLO) with
computer vision geometric analysis to detect objects, localize them with bounding
boxes, extract fine oriented contours, and compute calibrated real-world physical
dimensions (centimeters, millimeters, inches).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class YOLODetectionError(ValueError):
    """Raised when an error occurs during YOLO detection or measurement."""


@dataclass
class YOLODetection:
    """Structured record of a single YOLO-detected object with bounding and measurement metrics.

    Attributes:
        object_id: Sequential identifier (1-based).
        class_id: Numeric COCO class ID.
        class_name: Human-readable class name (e.g. 'bottle', 'cell phone', 'cup').
        confidence: Model prediction confidence score (0.0 to 1.0).
        bbox: Axis-aligned bounding box [x1, y1, x2, y2] in integer pixel coordinates.
        box_points: 4 corner coordinates (4x2 float array) of the oriented or standard rectangle.
        center: (x, y) centroid coordinates.
        length_px: Longest side dimension in pixels.
        width_px: Shorter side dimension in pixels.
        area_px2: Enclosed bounding area in pixels squared.
        angle_deg: Rotation angle in degrees (-90 to 90). 0 for axis-aligned boxes.
        is_calibrated: Whether physical scale factor was applied.
        length_cm: Physical length in centimeters (or None if uncalibrated).
        width_cm: Physical width in centimeters (or None if uncalibrated).
        length_mm: Physical length in millimeters (or None if uncalibrated).
        width_mm: Physical width in millimeters (or None if uncalibrated).
        length_in: Physical length in inches (or None if uncalibrated).
        width_in: Physical width in inches (or None if uncalibrated).
        area_cm2: Physical area in cm² (or None if uncalibrated).
        pixels_per_cm: Scale factor used for calibration.
        is_oriented: True if fine contour rotated minAreaRect was computed.
    """

    object_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]
    box_points: np.ndarray
    center: Tuple[float, float]
    length_px: float
    width_px: float
    area_px2: float
    angle_deg: float = 0.0
    is_calibrated: bool = False
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    length_in: Optional[float] = None
    width_in: Optional[float] = None
    area_cm2: Optional[float] = None
    pixels_per_cm: Optional[float] = None
    is_oriented: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection record to serializable dictionary."""
        return {
            "object_id": self.object_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 3),
            "bbox": [int(v) for v in self.bbox],
            "box_points": self.box_points.tolist() if self.box_points is not None else [],
            "center": [round(float(c), 2) for c in self.center],
            "length_px": round(float(self.length_px), 2),
            "width_px": round(float(self.width_px), 2),
            "area_px2": round(float(self.area_px2), 2),
            "angle_deg": round(float(self.angle_deg), 1),
            "is_calibrated": bool(self.is_calibrated),
            "length_cm": round(float(self.length_cm), 3) if self.length_cm is not None else None,
            "width_cm": round(float(self.width_cm), 3) if self.width_cm is not None else None,
            "length_mm": round(float(self.length_mm), 2) if self.length_mm is not None else None,
            "width_mm": round(float(self.width_mm), 2) if self.width_mm is not None else None,
            "length_in": round(float(self.length_in), 3) if self.length_in is not None else None,
            "width_in": round(float(self.width_in), 3) if self.width_in is not None else None,
            "area_cm2": round(float(self.area_cm2), 3) if self.area_cm2 is not None else None,
            "pixels_per_cm": round(float(self.pixels_per_cm), 3) if self.pixels_per_cm is not None else None,
            "is_oriented": bool(self.is_oriented),
        }

    def format_dimension_string(self, unit: str = "cm") -> str:
        """Format dimensions into human-readable label."""
        u = unit.lower().strip()
        if not self.is_calibrated:
            return f"{self.length_px:.1f} x {self.width_px:.1f} px"

        if u == "mm" and self.length_mm is not None and self.width_mm is not None:
            return f"{self.length_mm:.1f} x {self.width_mm:.1f} mm"
        elif u in ("in", "inch", "inches") and self.length_in is not None and self.width_in is not None:
            return f"{self.length_in:.2f} x {self.width_in:.2f} in"
        elif self.length_cm is not None and self.width_cm is not None:
            return f"{self.length_cm:.2f} x {self.width_cm:.2f} cm"
        return f"{self.length_px:.1f} x {self.width_px:.1f} px"


class YOLOMeasurementEngine:
    """Manages YOLO model inference, bounding box refinement, and physical dimension calculation."""

    def __init__(self, model_name: str = "yolov8m.pt") -> None:
        """Initialize the YOLO measurement engine.

        Args:
            model_name: Path or identifier of the YOLO model weights (e.g. yolov8m.pt, yolov8x.pt).
        """
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Lazy-load or initialize YOLO model weights."""
        from ultralytics import YOLO

        local_path = Path(self.model_name)
        root_path = Path(__file__).resolve().parent.parent / self.model_name
        models_dir_path = Path(__file__).resolve().parent.parent / "models" / self.model_name

        if local_path.is_file():
            self._model = YOLO(str(local_path))
        elif root_path.is_file():
            self._model = YOLO(str(root_path))
        elif models_dir_path.is_file():
            self._model = YOLO(str(models_dir_path))
        else:
            self._model = YOLO(self.model_name)

    @property
    def model(self):
        """Return the underlying YOLO model instance."""
        if self._model is None:
            self._load_model()
        return self._model

    def switch_model(self, model_name: str) -> None:
        """Switch to another YOLO model variant (e.g. yolov8x.pt, yolov8m.pt, yolov8s.pt)."""
        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self.model_name = model_name

    def detect_and_measure(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        pixels_per_cm: Optional[float] = None,
        mode: str = "oriented",  # 'oriented' or 'standard'
        target_classes: Optional[List[str]] = None,
    ) -> List[YOLODetection]:
        """Detect objects using YOLO and compute their geometric and physical dimensions.

        Args:
            image_bgr: Input image in OpenCV BGR format.
            conf_threshold: Minimum confidence threshold (0.05 to 1.0).
            iou_threshold: NMS IoU threshold.
            pixels_per_cm: Optional scale factor for physical unit conversion.
            mode: 'oriented' (fine rotated bounding box via contour inside bbox)
                  or 'standard' (axis-aligned YOLO bounding box).
            target_classes: Optional list of class names to filter by (case-insensitive).

        Returns:
            List of YOLODetection records for each detected object.
        """
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise YOLODetectionError("Input image must be a non-empty NumPy array.")

        h, w = image_bgr.shape[:2]

        # Run inference
        results = self.model(
            image_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections: List[YOLODetection] = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names_map = first_res.names

        target_classes_norm = (
            {c.strip().lower() for c in target_classes} if target_classes else None
        )

        object_id = 1
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0].item())
            class_name = names_map.get(cls_id, f"class_{cls_id}")
            confidence = float(box.conf[0].item())

            if target_classes_norm and class_name.lower() not in target_classes_norm:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1 = max(0, int(round(xyxy[0])))
            y1 = max(0, int(round(xyxy[1])))
            x2 = min(w, int(round(xyxy[2])))
            y2 = min(h, int(round(xyxy[3])))

            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)

            # Determine whether to compute oriented bounding box
            is_oriented = False
            box_points = None
            angle_deg = 0.0

            if mode == "oriented" and bbox_w > 8 and bbox_h > 8:
                roi = image_bgr[y1:y2, x1:x2]
                oriented_res = self._compute_oriented_box_in_roi(roi, x1, y1)
                if oriented_res is not None:
                    len_px, wid_px, center_pt, angle_deg, box_points = oriented_res
                    is_oriented = True
                else:
                    len_px = float(max(bbox_w, bbox_h))
                    wid_px = float(min(bbox_w, bbox_h))
                    center_pt = (float(x1 + bbox_w / 2.0), float(y1 + bbox_h / 2.0))
                    box_points = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                    )
            else:
                len_px = float(max(bbox_w, bbox_h))
                wid_px = float(min(bbox_w, bbox_h))
                center_pt = (float(x1 + bbox_w / 2.0), float(y1 + bbox_h / 2.0))
                box_points = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )

            area_px2 = float(len_px * wid_px)

            # Physical calibration calculation
            is_calibrated = False
            length_cm = None
            width_cm = None
            length_mm = None
            width_mm = None
            length_in = None
            width_in = None
            area_cm2 = None

            if pixels_per_cm is not None and pixels_per_cm > 0.0:
                is_calibrated = True
                length_cm = len_px / pixels_per_cm
                width_cm = wid_px / pixels_per_cm
                length_mm = length_cm * 10.0
                width_mm = width_cm * 10.0
                length_in = length_cm / 2.54
                width_in = width_cm / 2.54
                area_cm2 = length_cm * width_cm

            detection = YOLODetection(
                object_id=object_id,
                class_id=cls_id,
                class_name=class_name,
                confidence=confidence,
                bbox=[x1, y1, x2, y2],
                box_points=box_points,
                center=center_pt,
                length_px=len_px,
                width_px=wid_px,
                area_px2=area_px2,
                angle_deg=angle_deg,
                is_calibrated=is_calibrated,
                length_cm=length_cm,
                width_cm=width_cm,
                length_mm=length_mm,
                width_mm=width_mm,
                length_in=length_in,
                width_in=width_in,
                area_cm2=area_cm2,
                pixels_per_cm=pixels_per_cm,
                is_oriented=is_oriented,
            )
            detections.append(detection)
            object_id += 1

        return detections

    def _compute_oriented_box_in_roi(
        self, roi: np.ndarray, offset_x: int, offset_y: int
    ) -> Optional[Tuple[float, float, Tuple[float, float], float, np.ndarray]]:
        """Extract dominant object contour inside ROI and compute rotated minAreaRect with multi-stage fallback."""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            roi_area = roi.shape[0] * roi.shape[1]

            # Stage 1: Otsu's thresholding with morphological closing
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.count_nonzero(thresh) / thresh.size
            if white_ratio > 0.5:
                thresh = cv2.bitwise_not(thresh)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(
                thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidate_contour = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 0.08 * roi_area:
                    candidate_contour = largest

            # Stage 2: Canny edge contour fallback if Otsu gave insufficient coverage
            if candidate_contour is None:
                edges = cv2.Canny(blurred, 30, 120)
                dilated_edges = cv2.dilate(edges, kernel, iterations=2)
                edge_contours, _ = cv2.findContours(
                    dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if edge_contours:
                    largest_edge = max(edge_contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_edge) >= 0.08 * roi_area:
                        candidate_contour = largest_edge

            if candidate_contour is None:
                return None

            # Compute minAreaRect
            rect = cv2.minAreaRect(candidate_contour)
            (center_x, center_y), (rect_w, rect_h), angle = rect

            if rect_w <= 1 or rect_h <= 1:
                return None

            len_px = float(max(rect_w, rect_h))
            wid_px = float(min(rect_w, rect_h))

            # Transform back to full image coordinates
            full_center = (float(center_x + offset_x), float(center_y + offset_y))
            box_points = cv2.boxPoints(rect)
            box_points[:, 0] += offset_x
            box_points[:, 1] += offset_y

            return len_px, wid_px, full_center, float(angle), box_points
        except Exception:
            return None

    def annotate_detections(
        self,
        image_bgr: np.ndarray,
        detections: List[YOLODetection],
        unit: str = "cm",
        show_dimensions: bool = True,
        show_labels: bool = True,
        show_dimension_lines: bool = True,
        show_watermark: bool = True,
        palette_mode: str = "dynamic",  # 'dynamic' (class color) or 'cyan'
    ) -> np.ndarray:
        """Render high-clarity bounding boxes, dimensions, and badges on an image copy.

        Args:
            image_bgr: Input BGR image.
            detections: List of YOLODetection items.
            unit: Preferred unit ('cm', 'mm', 'in').
            show_dimensions: Whether to render dimension measurements.
            show_labels: Whether to render class and confidence badges.
            show_dimension_lines: Whether to draw midpoint dimension indicator lines.
            show_watermark: Whether to show calibration scale badge in top corner.
            palette_mode: Color scheme for bounding boxes.

        Returns:
            Annotated BGR image copy (original image remains pristine).
        """
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        annotated = image_bgr.copy()
        h, w = annotated.shape[:2]

        # Distinct modern color palette for classes
        PALETTE = [
            (244, 114, 182),  # pink / magenta
            (56, 189, 248),   # cyan
            (52, 211, 153),   # emerald
            (251, 191, 36),   # amber
            (167, 139, 250),  # purple
            (248, 113, 113),  # red-coral
            (96, 165, 250),   # light blue
            (74, 222, 128),   # lime green
        ]

        for det in detections:
            color = (
                PALETTE[det.class_id % len(PALETTE)]
                if palette_mode == "dynamic"
                else (0, 255, 255)
            )

            # Draw polygon or box
            if det.box_points is not None and len(det.box_points) == 4:
                pts = np.int32(det.box_points)
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

                # Draw corner vertices
                for pt in pts:
                    cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, color, -1, lineType=cv2.LINE_AA)

                # Draw dimension indicator lines along sides if enabled
                if show_dimension_lines:
                    self._draw_dimension_lines(annotated, pts, color)
            else:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Draw center point
            cx, cy = int(round(det.center[0])), int(round(det.center[1]))
            cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            # Construct badge text
            if show_labels:
                dim_str = det.format_dimension_string(unit) if show_dimensions else ""
                badge_title = f"#{det.object_id} {det.class_name} ({int(det.confidence * 100)}%)"
                badge_subtitle = f"Dim: {dim_str}" if show_dimensions else ""

                # Anchor badge near top-left of bbox
                bx = max(4, det.bbox[0])
                by = max(24, det.bbox[1] - 8)

                self._draw_badge(
                    annotated,
                    bx,
                    by,
                    title=badge_title,
                    subtitle=badge_subtitle,
                    accent_color=color,
                )

        # Show calibration watermark if scale is active
        if show_watermark and detections and detections[0].is_calibrated:
            scale_val = detections[0].pixels_per_cm
            scale_text = f"Calibrated Scale: {scale_val:.2f} px/cm [{unit}]"
            self._draw_watermark_badge(annotated, scale_text)

        return annotated

    def _draw_dimension_lines(self, img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int]) -> None:
        """Draw subtle midpoint markers and dimension indicator cues on polygon edges."""
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            cv2.circle(img, (mid_x, mid_y), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (mid_x, mid_y), 2, color, -1, lineType=cv2.LINE_AA)

    def _draw_badge(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        title: str,
        subtitle: str,
        accent_color: Tuple[int, int, int],
    ) -> None:
        """Render modern dark glass badge with accent color bar and readable typography."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.44
        font_thick = 1

        title_size, _ = cv2.getTextSize(title, font, font_scale, font_thick)
        sub_size = (0, 0)
        if subtitle:
            sub_size, _ = cv2.getTextSize(subtitle, font, font_scale - 0.04, font_thick)

        badge_w = max(title_size[0], sub_size[0]) + 16
        badge_h = title_size[1] + (sub_size[1] + 6 if subtitle else 0) + 12

        # Clamp badge inside image boundaries
        img_h, img_w = img.shape[:2]
        if x + badge_w > img_w - 4:
            x = max(4, img_w - badge_w - 4)
        if y - badge_h < 4:
            y = min(img_h - 4, y + badge_h + 10)

        top_left = (x, y - badge_h)
        bot_right = (x + badge_w, y)

        # Semi-transparent dark background
        sub_img = img[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            overlay = np.full_like(sub_img, 20)  # Dark slate
            cv2.addWeighted(overlay, 0.78, sub_img, 0.22, 0, sub_img)

        # Accent border & left stripe
        cv2.rectangle(img, top_left, bot_right, (60, 60, 60), 1, lineType=cv2.LINE_AA)
        cv2.line(
            img,
            (top_left[0], top_left[1]),
            (top_left[0], bot_right[1]),
            accent_color,
            3,
            lineType=cv2.LINE_AA,
        )

        # Text rendering
        t_pos_y = top_left[1] + title_size[1] + 4
        cv2.putText(
            img,
            title,
            (top_left[0] + 8, t_pos_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thick,
            lineType=cv2.LINE_AA,
        )

        if subtitle:
            s_pos_y = t_pos_y + sub_size[1] + 6
            cv2.putText(
                img,
                subtitle,
                (top_left[0] + 8, s_pos_y),
                font,
                font_scale - 0.04,
                accent_color,
                font_thick,
                lineType=cv2.LINE_AA,
            )

    def _draw_watermark_badge(self, img: np.ndarray, text: str) -> None:
        """Render top-left status pill showing active calibration scale."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

        pad = 8
        pill_w = tw + pad * 2 + 12
        pill_h = th + pad * 2

        x1, y1 = 16, 16
        x2, y2 = x1 + pill_w, y1 + pill_h

        img_h, img_w = img.shape[:2]
        if y2 < img_h and x2 < img_w:
            sub = img[y1:y2, x1:x2]
            overlay = np.full_like(sub, 24)
            cv2.addWeighted(overlay, 0.82, sub, 0.18, 0, sub)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 130), 1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x1 + pad + 2, y1 + pill_h // 2), 4, (0, 220, 130), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                img,
                text,
                (x1 + pad + 12, y1 + th + pad - 2),
                font,
                font_scale,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

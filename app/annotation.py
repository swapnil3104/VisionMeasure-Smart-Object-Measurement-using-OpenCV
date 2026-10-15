"""Annotation module for VisionMeasure.

Presents clear measurements on image copies by rendering rotated bounding rectangles,
dimension labels, and unit indicators while strictly leaving the original input image
unmodified and distinguishing calibrated physical dimensions from uncalibrated pixel metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import cv2
import numpy as np

from models.measurement_result import MeasurementResult

if TYPE_CHECKING:
    from app.contour_detection import ContourGeometry


class AnnotationError(ValueError):
    """Raised when an error occurs during image annotation or parameter validation."""


@dataclass
class AnnotationConfig:
    """Visual styling and configuration parameters for measurement annotations.

    Attributes:
        box_color: BGR color tuple for the rotated bounding rectangle (default: vibrant green).
        box_thickness: Line thickness in pixels for the bounding rectangle.
        text_color: BGR color tuple for label text (default: white).
        bg_color: BGR color tuple for label background badge (default: dark slate).
        bg_alpha: Transparency factor for label background badge (0.0 transparent to 1.0 opaque).
        font_face: OpenCV font face identifier (default: FONT_HERSHEY_SIMPLEX).
        font_scale: Font size scale factor.
        font_thickness: Stroke thickness for font glyphs.
        marker_color: BGR color tuple for edge midpoints and corners (default: amber/orange).
        marker_radius: Radius in pixels for midpoint markers.
        show_midpoints: Whether to render midpoint indicator markers on edges.
        show_empty_message: Whether to render a status badge when no measurements are present.
        empty_message: Message text rendered when measurements list is empty.
        preferred_unit: Preferred physical unit ('cm' or 'mm'). Ignored for uncalibrated results.
    """

    box_color: Tuple[int, int, int] = (0, 255, 0)
    box_thickness: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)
    bg_color: Tuple[int, int, int] = (30, 30, 30)
    bg_alpha: float = 0.75
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.5
    font_thickness: int = 1
    marker_color: Tuple[int, int, int] = (0, 165, 255)
    marker_radius: int = 3
    show_midpoints: bool = True
    show_empty_message: bool = True
    empty_message: str = "No objects detected"
    preferred_unit: str = "cm"


def _validate_image_array(image: np.ndarray) -> np.ndarray:
    """Validate that the input is a readable NumPy image array.

    Raises:
        AnnotationError: If image is None, empty, or not uint8.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise AnnotationError("Input image must be a valid NumPy ndarray.")

    if image.size == 0:
        raise AnnotationError("Input image array is empty.")

    if image.dtype != np.uint8:
        raise AnnotationError(
            f"Input image array must have dtype uint8, got {image.dtype}."
        )

    if image.ndim not in (2, 3):
        raise AnnotationError(
            f"Input image must be 2D (grayscale) or 3D (color), got {image.ndim} dimensions."
        )

    return image


def _prepare_canvas(image: np.ndarray) -> np.ndarray:
    """Create a separate 3-channel BGR copy of the input image.

    Guarantees the input array is never modified in-place.
    """
    _validate_image_array(image)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        return image.copy()


def format_annotation_text(
    measurement: MeasurementResult,
    preferred_unit: str = "cm",
    include_id: bool = True,
) -> str:
    """Generate display text for an object's measurement metrics.

    Strictly ensures uncalibrated results are labeled in pixels with an explicit
    '(Uncalibrated)' marker and never presented with physical units (cm or mm).

    Args:
        measurement: The MeasurementResult data record.
        preferred_unit: Physical unit for calibrated measurements ('cm' or 'mm').
        include_id: Whether to prepend the object ID (e.g. 'Obj 1: ').

    Returns:
        Formatted label string.
    """
    if not isinstance(measurement, MeasurementResult):
        raise AnnotationError(
            f"Expected MeasurementResult instance, got {type(measurement).__name__}."
        )

    return measurement.format_dimension_string(unit=preferred_unit, include_id=include_id)


def calculate_label_position(
    box_points: np.ndarray,
    text_size: Tuple[int, int],
    image_shape: Tuple[int, ...],
    padding: int = 4,
    offset_y: int = 8,
) -> Tuple[int, int]:
    """Calculate label origin coordinates to avoid covering the object contour.

    Places the badge above the highest point of the bounding box where space permits;
    if near the top image boundary, repositions below the lowest bounding point or
    safely clamps within image boundaries.

    Args:
        box_points: Array of 4 corner coordinates (shape: (4, 2)).
        text_size: (width, height) in pixels of the rendered text string.
        image_shape: Shape tuple of the target image canvas (height, width, ...).
        padding: Internal margin padding inside the background badge.
        offset_y: Clearance gap between bounding box and badge.

    Returns:
        (x, y) coordinates for bottom-left text baseline inside the badge.
    """
    img_h, img_w = image_shape[0], image_shape[1]
    text_w, text_h = text_size

    badge_w = text_w + 2 * padding
    badge_h = text_h + 2 * padding

    min_y = float(np.min(box_points[:, 1]))
    max_y = float(np.max(box_points[:, 1]))

    # Top-most corner point
    top_pt_idx = int(np.argmin(box_points[:, 1]))
    top_x = float(box_points[top_pt_idx, 0])

    # Center horizontally on top vertex, or center of box
    badge_x = int(round(top_x - badge_w / 2.0))

    # Try placing above top boundary
    badge_y = int(round(min_y - badge_h - offset_y))

    if badge_y < 4:
        # Not enough room above, place below lowest bounding corner
        badge_y = int(round(max_y + offset_y))
        if badge_y + badge_h > img_h - 4:
            # Not enough room below either, clamp inside the canvas
            badge_y = max(4, min(img_h - badge_h - 4, int(round(min_y + 4))))

    # Clamp horizontally inside canvas
    badge_x = max(4, min(img_w - badge_w - 4, badge_x))

    # Text baseline coordinates (OpenCV putText uses bottom-left of text baseline)
    text_x = badge_x + padding
    text_y = badge_y + padding + text_h

    return text_x, text_y


def draw_label_badge(
    canvas: np.ndarray,
    text: str,
    text_origin: Tuple[int, int],
    text_size: Tuple[int, int],
    baseline: int,
    config: AnnotationConfig,
    padding: int = 4,
) -> None:
    """Render text with a contrasting background badge on the canvas.

    Args:
        canvas: Image array to draw on (modified in-place).
        text: String label to display.
        text_origin: (text_x, text_y) bottom-left origin for cv2.putText.
        text_size: (width, height) of text.
        baseline: Baseline offset returned by cv2.getTextSize.
        config: AnnotationConfig styling parameters.
        padding: Padding margin in pixels around text.
    """
    text_x, text_y = text_origin
    text_w, text_h = text_size

    box_x1 = max(0, text_x - padding)
    box_y1 = max(0, text_y - text_h - padding)
    box_x2 = min(canvas.shape[1] - 1, text_x + text_w + padding)
    box_y2 = min(canvas.shape[0] - 1, text_y + baseline + padding)

    if config.bg_alpha >= 1.0:
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), config.bg_color, -1)
    elif config.bg_alpha > 0.0:
        sub_img = canvas[box_y1:box_y2, box_x1:box_x2]
        if sub_img.size > 0:
            bg_rect = np.full(sub_img.shape, config.bg_color, dtype=np.uint8)
            blended = cv2.addWeighted(bg_rect, config.bg_alpha, sub_img, 1.0 - config.bg_alpha, 0)
            canvas[box_y1:box_y2, box_x1:box_x2] = blended

    # Draw border around badge
    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), config.box_color, 1)

    # Draw text
    cv2.putText(
        canvas,
        text,
        (text_x, text_y),
        config.font_face,
        config.font_scale,
        config.text_color,
        config.font_thickness,
        cv2.LINE_AA,
    )


def draw_rotated_box(
    canvas: np.ndarray,
    box_points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw a 4-point rotated bounding rectangle on the canvas.

    Args:
        canvas: Image array to draw on.
        box_points: Array of 4 corner coordinates (shape: (4, 2)).
        color: BGR color tuple.
        thickness: Line stroke thickness.
    """
    if box_points is None or not isinstance(box_points, np.ndarray) or box_points.shape[0] != 4:
        raise AnnotationError("box_points must be a NumPy array of shape (4, 2).")

    int_box = np.round(box_points).astype(np.int32)
    cv2.drawContours(canvas, [int_box], 0, color, thickness)


def draw_edge_midpoints(
    canvas: np.ndarray,
    box_points: np.ndarray,
    color: Tuple[int, int, int] = (0, 165, 255),
    radius: int = 3,
) -> None:
    """Draw indicator circles at the midpoints of the 4 edges of a bounding box.

    Args:
        canvas: Image array to draw on.
        box_points: Array of 4 corner coordinates.
        color: BGR color tuple.
        radius: Circle marker radius in pixels.
    """
    if box_points is None or box_points.shape[0] != 4:
        return

    for i in range(4):
        pt1 = box_points[i]
        pt2 = box_points[(i + 1) % 4]
        mid_x = int(round((pt1[0] + pt2[0]) / 2.0))
        mid_y = int(round((pt1[1] + pt2[1]) / 2.0))
        cv2.circle(canvas, (mid_x, mid_y), radius, color, -1)


def draw_empty_state_badge(
    canvas: np.ndarray,
    message: str = "No objects detected",
    config: Optional[AnnotationConfig] = None,
) -> None:
    """Draw a clean status badge indicating an empty detection state.

    Args:
        canvas: Target image array (modified in-place).
        message: Notification message text.
        config: Optional AnnotationConfig.
    """
    cfg = config or AnnotationConfig()
    text_size, baseline = cv2.getTextSize(
        message, cfg.font_face, cfg.font_scale, cfg.font_thickness
    )
    img_h, img_w = canvas.shape[:2]

    # Position at top center
    badge_w = text_size[0] + 16
    badge_h = text_size[1] + 12

    x1 = max(4, (img_w - badge_w) // 2)
    y1 = 12
    x2 = min(img_w - 4, x1 + badge_w)
    y2 = y1 + badge_h

    sub_img = canvas[y1:y2, x1:x2]
    if sub_img.size > 0:
        bg_rect = np.full(sub_img.shape, (40, 40, 40), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(bg_rect, 0.85, sub_img, 0.15, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 100, 100), 1)

    text_x = x1 + 8
    text_y = y1 + text_size[1] + 6
    cv2.putText(
        canvas,
        message,
        (text_x, text_y),
        cfg.font_face,
        cfg.font_scale,
        (220, 220, 220),
        cfg.font_thickness,
        cv2.LINE_AA,
    )


def annotate_single_object(
    canvas: np.ndarray,
    measurement: MeasurementResult,
    box_points: Optional[np.ndarray] = None,
    config: Optional[AnnotationConfig] = None,
    preferred_unit: str = "cm",
) -> None:
    """Annotate a single measured object on the provided canvas.

    Args:
        canvas: Image array to annotate (modified in-place).
        measurement: MeasurementResult record containing dimensional metrics.
        box_points: Optional 4-corner coordinates. If None, uses measurement.box_points.
        config: Optional AnnotationConfig styling configuration.
        preferred_unit: Preferred physical unit ('cm' or 'mm').

    Raises:
        AnnotationError: If bounding box coordinates cannot be found.
    """
    cfg = config or AnnotationConfig()
    pts = box_points if box_points is not None else measurement.box_points

    if pts is None:
        raise AnnotationError(
            f"Object {measurement.object_id} does not have bounding box points available."
        )

    if not isinstance(pts, np.ndarray) or pts.shape[0] != 4:
        raise AnnotationError(
            f"Invalid box_points for object {measurement.object_id}: expected shape (4, 2)."
        )

    # 1. Draw rotated bounding box
    draw_rotated_box(canvas, pts, color=cfg.box_color, thickness=cfg.box_thickness)

    # 2. Draw edge midpoints if enabled
    if cfg.show_midpoints:
        draw_edge_midpoints(canvas, pts, color=cfg.marker_color, radius=cfg.marker_radius)

    # 3. Format measurement label
    label_text = format_annotation_text(
        measurement=measurement,
        preferred_unit=preferred_unit or cfg.preferred_unit,
        include_id=True,
    )

    # 4. Compute label dimensions and optimal position
    text_size, baseline = cv2.getTextSize(
        label_text, cfg.font_face, cfg.font_scale, cfg.font_thickness
    )
    text_origin = calculate_label_position(
        box_points=pts,
        text_size=text_size,
        image_shape=canvas.shape,
    )

    # 5. Render label badge
    draw_label_badge(
        canvas=canvas,
        text=label_text,
        text_origin=text_origin,
        text_size=text_size,
        baseline=baseline,
        config=cfg,
    )


def annotate_measurements(
    image: np.ndarray,
    measurements: Sequence[MeasurementResult],
    geometries: Optional[Sequence["ContourGeometry"]] = None,
    config: Optional[AnnotationConfig] = None,
    preferred_unit: str = "cm",
) -> np.ndarray:
    """Create an annotated copy of the source image displaying object measurements.

    Guarantees:
    - The original input image is never modified.
    - An empty measurements sequence produces a valid image copy and an explicit no-results state.
    - Rotated bounding rectangles and dimensional metrics are clearly rendered.
    - Physical units ('cm' or 'mm') are displayed ONLY when calibration is valid.
    - Uncalibrated results are explicitly displayed in pixel units with an '(Uncalibrated)' marker.

    Args:
        image: Source image array (uint8, grayscale or BGR).
        measurements: Sequence of MeasurementResult records to annotate.
        geometries: Optional sequence of ContourGeometry corresponding to measurements.
        config: Optional AnnotationConfig styling options.
        preferred_unit: Preferred physical unit ('cm' or 'mm', default: 'cm').

    Returns:
        New BGR NumPy image array containing the rendered annotations.

    Raises:
        AnnotationError: If image array is invalid or parameter types are incorrect.
    """
    canvas = _prepare_canvas(image)
    cfg = config or AnnotationConfig()

    if not measurements:
        if cfg.show_empty_message:
            draw_empty_state_badge(canvas, message=cfg.empty_message, config=cfg)
        return canvas

    # Map geometries by index if provided
    geom_list = list(geometries) if geometries is not None else None

    for idx, meas in enumerate(measurements):
        if not isinstance(meas, MeasurementResult):
            raise AnnotationError(
                f"Item at index {idx} is not a MeasurementResult instance, got {type(meas).__name__}."
            )

        box_pts = meas.box_points
        if box_pts is None and geom_list is not None and idx < len(geom_list):
            box_pts = geom_list[idx].box_points

        annotate_single_object(
            canvas=canvas,
            measurement=meas,
            box_points=box_pts,
            config=cfg,
            preferred_unit=preferred_unit,
        )

    return canvas

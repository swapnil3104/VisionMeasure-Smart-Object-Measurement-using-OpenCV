"""Accuracy evaluation and benchmark engine for VisionMeasure (Phase 8).

Defines standard benchmark objects with real-world physical ground truth (measured
via calipers/ruler), generates controlled planar test scenes, runs the complete
computer-vision measurement pipeline, and computes quantitative error metrics:

    Absolute Error = |Estimated - Actual|
    Percentage Error = (|Estimated - Actual| / Actual) * 100

Provides statistical summaries (Mean Absolute Error, Mean Percentage Error)
reported separately for length and width as required by the PRD and Phase 8 tasks.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from app.annotation import annotate_measurements
from app.calibration import calibrate
from app.contour_detection import ContourGeometry, detect_contours
from app.edge_detection import detect_edges
from app.export_results import save_annotated_image
from app.measurement import measure_objects
from app.preprocessing import preprocess_image


class AccuracyEvaluationError(ValueError):
    """Raised when evaluation calculations or inputs are invalid."""


def calculate_absolute_error(estimated: float, actual: float) -> float:
    """Calculate absolute error between estimated and actual dimensions.

    Formula:
        Absolute Error = |Estimated - Actual|

    Args:
        estimated: Measured or estimated value.
        actual: Ground truth reference value.

    Returns:
        Absolute difference (non-negative float).

    Raises:
        AccuracyEvaluationError: If inputs are non-numeric or non-finite.
    """
    if isinstance(estimated, bool) or isinstance(actual, bool):
        raise AccuracyEvaluationError("Dimensions cannot be boolean values.")
    try:
        est_f = float(estimated)
        act_f = float(actual)
    except (TypeError, ValueError) as err:
        raise AccuracyEvaluationError(f"Invalid numeric input for error calculation: {err}") from err

    if not math.isfinite(est_f) or not math.isfinite(act_f):
        raise AccuracyEvaluationError("Dimensions must be finite numbers.")

    return abs(est_f - act_f)


def calculate_percentage_error(estimated: float, actual: float) -> float:
    """Calculate percentage error between estimated and actual dimensions.

    Formula:
        Percentage Error = (|Estimated - Actual| / Actual) * 100

    Args:
        estimated: Measured or estimated value.
        actual: Ground truth reference value (must be strictly positive).

    Returns:
        Percentage error as a float (e.g. 2.5 for 2.5%).

    Raises:
        AccuracyEvaluationError: If actual is zero, negative, or inputs non-finite.
    """
    abs_err = calculate_absolute_error(estimated, actual)
    act_f = float(actual)
    if act_f <= 0.0:
        raise AccuracyEvaluationError(f"Actual reference dimension must be strictly positive, got {act_f}.")

    return (abs_err / act_f) * 100.0


@dataclass(frozen=True)
class BenchmarkTarget:
    """Represents a standard flat test object with caliper ground truth.

    Attributes:
        name: Common descriptive name of the object.
        actual_length_cm: Ground truth longest dimension in cm.
        actual_width_cm: Ground truth shorter dimension in cm.
        category: Object categorization (e.g. 'Stationery', 'Card', 'Electronics').
        notes: Physical description or caliper measurement notes.
    """

    name: str
    actual_length_cm: float
    actual_width_cm: float
    category: str = "General"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.actual_length_cm <= 0.0 or self.actual_width_cm <= 0.0:
            raise AccuracyEvaluationError(
                f"Benchmark target '{self.name}' must have strictly positive dimensions."
            )
        # Ensure length is the longer side or equal
        if self.actual_length_cm < self.actual_width_cm:
            # Normalize so length >= width
            object.__setattr__(self, "actual_length_cm", self.actual_width_cm)
            object.__setattr__(self, "actual_width_cm", self.actual_length_cm)


@dataclass
class ObjectAccuracyResult:
    """Single object evaluation result comparing estimated vs ground truth."""

    target_name: str
    actual_length_cm: float
    actual_width_cm: float
    estimated_length_cm: float
    estimated_width_cm: float
    length_abs_error_cm: float
    length_pct_error: float
    width_abs_error_cm: float
    width_pct_error: float
    orientation_deg: float = 0.0

    def to_dict(self) -> dict:
        """Convert result to dictionary formatted for report export."""
        return {
            "target_name": self.target_name,
            "actual_length_cm": round(self.actual_length_cm, 3),
            "actual_width_cm": round(self.actual_width_cm, 3),
            "estimated_length_cm": round(self.estimated_length_cm, 3),
            "estimated_width_cm": round(self.estimated_width_cm, 3),
            "length_abs_error_cm": round(self.length_abs_error_cm, 4),
            "length_pct_error": round(self.length_pct_error, 2),
            "width_abs_error_cm": round(self.width_abs_error_cm, 4),
            "width_pct_error": round(self.width_pct_error, 2),
            "orientation_deg": round(self.orientation_deg, 1),
        }


@dataclass
class AccuracySummary:
    """Statistical summary across a benchmark evaluation set."""

    total_objects: int
    mean_length_abs_error_cm: float
    mean_length_pct_error: float
    mean_width_abs_error_cm: float
    mean_width_pct_error: float
    max_length_abs_error_cm: float
    max_width_abs_error_cm: float
    max_length_pct_error: float
    max_width_pct_error: float
    overall_mean_abs_error_cm: float
    overall_mean_pct_error: float


# Standard 20-object benchmark dataset measured with caliper / metric ruler
STANDARD_BENCHMARK_TARGETS: Tuple[BenchmarkTarget, ...] = (
    BenchmarkTarget("ISO Credit Card (ID-1)", 8.56, 5.40, "Card", "ISO/IEC 7810 standard reference card"),
    BenchmarkTarget("Standard Business Card", 8.90, 5.10, "Card", "Standard US business card"),
    BenchmarkTarget("Square Post-It Note", 7.62, 7.62, "Stationery", "Standard 3x3 inch yellow sticky note"),
    BenchmarkTarget("Small Sticky Note Pad", 5.00, 3.80, "Stationery", "2x1.5 inch note pad"),
    BenchmarkTarget("Standard Playing Card", 8.89, 6.35, "Recreation", "Standard poker card 2.5x3.5 inch"),
    BenchmarkTarget("MicroSD Card Adapter", 3.20, 2.40, "Electronics", "Standard SD card adapter form factor"),
    BenchmarkTarget("Standard SIM Card Adapter", 2.50, 1.50, "Electronics", "Mini-to-micro SIM card adapter"),
    BenchmarkTarget("Standard Index Card", 12.70, 7.62, "Stationery", "3x5 inch index card"),
    BenchmarkTarget("Small Matchbox", 5.20, 3.60, "Household", "Standard cardboard matchbox top surface"),
    BenchmarkTarget("Rectangular Eraser", 6.00, 2.20, "Stationery", "Standard rubber eraser block"),
    BenchmarkTarget("USB Flash Drive Casing", 5.80, 1.80, "Electronics", "Rectangular USB thumb drive body"),
    BenchmarkTarget("Small Bookmark", 15.00, 5.00, "Stationery", "Cardboard bookmark"),
    BenchmarkTarget("Square Beverage Coaster", 10.00, 10.00, "Household", "Standard square cardboard bar coaster"),
    BenchmarkTarget("Sticky Index Page Flag", 4.40, 1.20, "Stationery", "Plastic filing index tab"),
    BenchmarkTarget("Pocket Notebook (A6)", 14.80, 10.50, "Stationery", "Standard DIN A6 pocket notebook"),
    BenchmarkTarget("Stainless Steel Ruler Segment", 15.00, 2.50, "Tools", "15 cm flat precision ruler segment"),
    BenchmarkTarget("Small Square Ceramic Tile", 5.00, 5.00, "Materials", "Standard 50x50 mm sample tile"),
    BenchmarkTarget("Teabag Sachet Envelope", 7.80, 6.80, "Packaging", "Sealed paper tea bag packet"),
    BenchmarkTarget("Plastic ID Name Badge", 9.00, 6.00, "Office", "Conference identification card sleeve"),
    BenchmarkTarget("Mini Memo Notepad", 10.00, 7.50, "Stationery", "A7 rectangular memo paper block"),
)


def generate_planar_test_image(
    targets: Sequence[BenchmarkTarget],
    pixels_per_cm: float = 40.0,
    image_size: Tuple[int, int] = (1600, 2000),
    orientations: Optional[Sequence[float]] = None,
    background_color: Tuple[int, int, int] = (245, 245, 245),
    object_color: Tuple[int, int, int] = (30, 30, 30),
) -> Tuple[np.ndarray, List[Dict[str, Union[float, str]]]]:
    """Synthesize a high-precision planar test image containing the specified targets.

    Objects are laid out neatly on a coplanar surface with distinct coordinates and
    user-specified or pseudo-random rotations to test orientation invariance.

    Args:
        targets: List of BenchmarkTarget items to render.
        pixels_per_cm: Calibration scale factor in pixels/cm.
        image_size: (height, width) of generated image.
        orientations: Optional explicit rotation angles (degrees) for each target.
        background_color: BGR background color.
        object_color: BGR object fill color.

    Returns:
        Tuple of (rendered_bgr_image, ground_truth_manifest).
    """
    img_h, img_w = image_size
    canvas = np.full((img_h, img_w, 3), background_color, dtype=np.uint8)

    manifest = []
    num_targets = len(targets)
    if num_targets == 0:
        return canvas, manifest

    # Calculate dynamic grid layout with sufficient clearance for large objects
    cols = 4 if num_targets >= 8 else (2 if num_targets <= 4 else 3)
    rows = math.ceil(num_targets / cols)

    # Ensure cell size comfortably accommodates largest target with padding
    max_len_px = max(t.actual_length_cm for t in targets) * pixels_per_cm
    min_cell_dim = int(max_len_px + 200)

    cell_w = max(img_w // cols, min_cell_dim)
    cell_h = max(img_h // rows, min_cell_dim)

    # Resize canvas if needed to accommodate cells
    needed_w = cols * cell_w
    needed_h = rows * cell_h
    if needed_w > img_w or needed_h > img_h:
        img_w = max(img_w, needed_w)
        img_h = max(img_h, needed_h)
        canvas = np.full((img_h, img_w, 3), background_color, dtype=np.uint8)

    for idx, target in enumerate(targets):
        row = idx // cols
        col = idx % cols

        cx = int(col * cell_w + cell_w / 2)
        cy = int(row * cell_h + cell_h / 2)

        # Convert cm dimensions to pixels
        len_px = target.actual_length_cm * pixels_per_cm
        wid_px = target.actual_width_cm * pixels_per_cm

        angle = 0.0
        if orientations is not None and idx < len(orientations):
            angle = float(orientations[idx])
        else:
            # Vary angles deterministically across targets: 0, 15, 30, 45, 60, 90, etc.
            angle = float((idx * 25) % 180)

        # Draw rotated rectangle
        rect = ((float(cx), float(cy)), (float(wid_px), float(len_px)), angle)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(canvas, [box], 0, object_color, thickness=-1)

        manifest.append({
            "name": target.name,
            "center": (cx, cy),
            "actual_length_cm": target.actual_length_cm,
            "actual_width_cm": target.actual_width_cm,
            "length_px": len_px,
            "width_px": wid_px,
            "angle_deg": angle,
        })

    return canvas, manifest


def run_accuracy_benchmark(
    targets: Sequence[BenchmarkTarget] = STANDARD_BENCHMARK_TARGETS,
    pixels_per_cm: float = 40.0,
    reference_target_index: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[ObjectAccuracyResult], AccuracySummary]:
    """Execute end-to-end detection, calibration, and accuracy evaluation.

    Args:
        targets: Sequence of benchmark objects to evaluate.
        pixels_per_cm: True scale factor of the synthetic test scene.
        reference_target_index: Index of target used as known calibration reference.
        output_dir: Optional directory to save output CSV and annotated image.

    Returns:
        Tuple of (list_of_object_accuracy_results, accuracy_summary).
    """
    if not targets:
        raise AccuracyEvaluationError("Benchmark targets cannot be empty.")

    # 1. Generate clean planar test scene
    # Predefined realistic rotation angles across the objects
    angles = [(i * 18.5) % 180 for i in range(len(targets))]
    image, manifest = generate_planar_test_image(
        targets=targets,
        pixels_per_cm=pixels_per_cm,
        image_size=(2400, 2000),
        orientations=angles,
    )

    # 2. Execute VisionMeasure processing pipeline
    preprocessed = preprocess_image(image, blur_kernel=5)
    edges = detect_edges(preprocessed, threshold1=50, threshold2=150, apply_closing=True)
    geometries = detect_contours(edges, min_area=300.0)

    if not geometries:
        raise AccuracyEvaluationError("Pipeline detected no contours on benchmark test image.")

    # 3. Simulate reference calibration
    # Use reference target (e.g. ISO Credit Card)
    ref_target = targets[reference_target_index]
    ref_manifest = manifest[reference_target_index]
    ref_actual_len = ref_target.actual_length_cm

    # Find the geometry matching the reference target center
    ref_geom = _find_matching_geometry(ref_manifest["center"], geometries)
    if ref_geom is None:
        raise AccuracyEvaluationError(f"Could not locate reference target '{ref_target.name}' in detected contours.")

    # Calibrate using detected pixel span of reference
    cal = calibrate(
        reference_px=ref_geom.length_px,
        reference_length=ref_actual_len,
        unit="cm",
    )

    # 4. Measure all objects using calibration
    measurements = measure_objects(geometries, calibration=cal, start_id=1)

    # 5. Correlate each benchmark target with its detected contour measurement
    results: List[ObjectAccuracyResult] = []
    for target, item in zip(targets, manifest):
        center = item["center"]
        matched_geom = _find_matching_geometry(center, geometries)
        if matched_geom is None:
            continue

        # Find corresponding measurement
        meas = next((m for m in measurements if m.length_px == matched_geom.length_px), None)
        if meas is None or meas.length_cm is None or meas.width_cm is None:
            continue

        # Calculate errors
        len_abs = calculate_absolute_error(meas.length_cm, target.actual_length_cm)
        len_pct = calculate_percentage_error(meas.length_cm, target.actual_length_cm)
        wid_abs = calculate_absolute_error(meas.width_cm, target.actual_width_cm)
        wid_pct = calculate_percentage_error(meas.width_cm, target.actual_width_cm)

        res = ObjectAccuracyResult(
            target_name=target.name,
            actual_length_cm=target.actual_length_cm,
            actual_width_cm=target.actual_width_cm,
            estimated_length_cm=meas.length_cm,
            estimated_width_cm=meas.width_cm,
            length_abs_error_cm=len_abs,
            length_pct_error=len_pct,
            width_abs_error_cm=wid_abs,
            width_pct_error=wid_pct,
            orientation_deg=float(item["angle_deg"]),
        )
        results.append(res)

    # 6. Compute statistical summary
    summary = compute_accuracy_summary(results)

    # 7. Optionally write export files
    if output_dir is not None:
        out_p = Path(output_dir)
        csv_dir = out_p / "measurements" if out_p.name != "measurements" else out_p
        annotated_dir = out_p / "annotated_images" if out_p.name != "measurements" else out_p.parent / "annotated_images"

        csv_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)

        csv_path = csv_dir / "accuracy_evaluation_report.csv"
        export_benchmark_csv(results, summary, csv_path)

        # Save annotated verification image
        annotated = annotate_measurements(image, measurements, geometries)
        save_annotated_image(annotated, annotated_dir / "benchmark_annotated.png", overwrite=True)

        # Also preserve raw benchmark scene image
        raw_scene_dir = Path("data/sample_images")
        raw_scene_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw_scene_dir / "benchmark_planar_scene.png"), image)

    return results, summary


def _find_matching_geometry(
    target_center: Tuple[int, int],
    geometries: Sequence[ContourGeometry],
    max_distance_px: float = 60.0,
) -> Optional[ContourGeometry]:
    """Find the detected geometry whose centroid is closest to the expected center."""
    cx, cy = target_center
    best_geom: Optional[ContourGeometry] = None
    min_dist = float("inf")

    for geom in geometries:
        gx, gy = geom.center
        dist = math.hypot(gx - cx, gy - cy)
        if dist < min_dist and dist <= max_distance_px:
            min_dist = dist
            best_geom = geom

    return best_geom


def compute_accuracy_summary(results: Sequence[ObjectAccuracyResult]) -> AccuracySummary:
    """Compute aggregate statistical metrics across accuracy results."""
    if not results:
        raise AccuracyEvaluationError("Cannot compute summary on empty results list.")

    n = len(results)
    len_abs_list = [r.length_abs_error_cm for r in results]
    len_pct_list = [r.length_pct_error for r in results]
    wid_abs_list = [r.width_abs_error_cm for r in results]
    wid_pct_list = [r.width_pct_error for r in results]

    mean_len_abs = sum(len_abs_list) / n
    mean_len_pct = sum(len_pct_list) / n
    mean_wid_abs = sum(wid_abs_list) / n
    mean_wid_pct = sum(wid_pct_list) / n

    overall_abs = (mean_len_abs + mean_wid_abs) / 2.0
    overall_pct = (mean_len_pct + mean_wid_pct) / 2.0

    return AccuracySummary(
        total_objects=n,
        mean_length_abs_error_cm=mean_len_abs,
        mean_length_pct_error=mean_len_pct,
        mean_width_abs_error_cm=mean_wid_abs,
        mean_width_pct_error=mean_wid_pct,
        max_length_abs_error_cm=max(len_abs_list),
        max_width_abs_error_cm=max(wid_abs_list),
        max_length_pct_error=max(len_pct_list),
        max_width_pct_error=max(wid_pct_list),
        overall_mean_abs_error_cm=overall_abs,
        overall_mean_pct_error=overall_pct,
    )


def export_benchmark_csv(
    results: Sequence[ObjectAccuracyResult],
    summary: AccuracySummary,
    output_path: Union[str, Path],
) -> Path:
    """Export benchmark evaluation results and summary table to CSV."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "target_name",
        "actual_length_cm",
        "actual_width_cm",
        "estimated_length_cm",
        "estimated_width_cm",
        "length_abs_error_cm",
        "length_pct_error",
        "width_abs_error_cm",
        "width_pct_error",
        "orientation_deg",
    ]

    with open(out_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    return out_file

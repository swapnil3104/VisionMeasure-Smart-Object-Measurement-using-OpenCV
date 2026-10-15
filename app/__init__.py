"""VisionMeasure application package."""

from app.annotation import (
    AnnotationConfig,
    AnnotationError,
    annotate_measurements,
    annotate_single_object,
    calculate_label_position,
    draw_edge_midpoints,
    draw_empty_state_badge,
    draw_label_badge,
    draw_rotated_box,
    format_annotation_text,
)
from app.calibration import (
    CalibrationData,
    CalibrationError,
    calculate_pixel_distance,
    calculate_pixels_per_unit,
    calibrate,
    calibrate_from_geometry,
    calibrate_from_points,
    normalize_unit,
)
from app.contour_detection import (
    ContourDetectionError,
    ContourGeometry,
    calculate_contour_geometry,
    detect_contours,
    extract_external_contours,
    filter_contours_by_area,
)
from app.edge_detection import (
    EdgeDetectionError,
    apply_morphological_closing,
    detect_edges,
)
from app.image_input import (
    CameraError,
    ImageInputError,
    ImageNotFoundError,
    ImageReadError,
    UnsupportedFormatError,
    capture_frame,
    load_image,
    validate_image_array,
)
from app.measurement import (
    MeasurementError,
    calculate_physical_dimension,
    measure_object,
    measure_objects,
    validate_geometry,
)
from app.export_results import (
    CSVExportError,
    ExportError,
    ImageExportError,
    export_csv,
    save_annotated_image,
)
from app.gui import (
    VisionMeasureGUI,
    run_gui,
)
from app.preprocessing import (
    PreprocessingError,
    apply_gaussian_blur,
    preprocess_image,
    to_grayscale,
)
from app.accuracy_benchmark import (
    AccuracyEvaluationError,
    AccuracySummary,
    BenchmarkTarget,
    ObjectAccuracyResult,
    STANDARD_BENCHMARK_TARGETS,
    calculate_absolute_error,
    calculate_percentage_error,
    compute_accuracy_summary,
    export_benchmark_csv,
    generate_planar_test_image,
    run_accuracy_benchmark,
)
from app.yolo_detector import (
    YOLODetection,
    YOLODetectionError,
    YOLOMeasurementEngine,
)
from app.web_server import run_server

__all__ = [
    # Image Input
    "load_image",
    "capture_frame",
    "validate_image_array",
    "ImageInputError",
    "ImageNotFoundError",
    "UnsupportedFormatError",
    "ImageReadError",
    "CameraError",
    # Preprocessing
    "to_grayscale",
    "apply_gaussian_blur",
    "preprocess_image",
    "PreprocessingError",
    # Edge Detection
    "detect_edges",
    "apply_morphological_closing",
    "EdgeDetectionError",
    # Contour Detection & Geometry
    "ContourGeometry",
    "ContourDetectionError",
    "extract_external_contours",
    "calculate_contour_geometry",
    "filter_contours_by_area",
    "detect_contours",
    # Calibration
    "CalibrationData",
    "CalibrationError",
    "normalize_unit",
    "calculate_pixels_per_unit",
    "calibrate",
    "calibrate_from_geometry",
    "calibrate_from_points",
    "calculate_pixel_distance",
    # Measurement
    "MeasurementError",
    "calculate_physical_dimension",
    "validate_geometry",
    "measure_object",
    "measure_objects",
    # Annotation
    "AnnotationConfig",
    "AnnotationError",
    "annotate_measurements",
    "annotate_single_object",
    "draw_rotated_box",
    "draw_label_badge",
    "draw_edge_midpoints",
    "draw_empty_state_badge",
    "calculate_label_position",
    "format_annotation_text",
    # Export
    "save_annotated_image",
    "export_csv",
    "ExportError",
    "ImageExportError",
    "CSVExportError",
    # GUI
    "VisionMeasureGUI",
    "run_gui",
    # YOLO & Web
    "YOLODetection",
    "YOLODetectionError",
    "YOLOMeasurementEngine",
    "run_server",
    # Accuracy Benchmark & Evaluation
    "AccuracyEvaluationError",
    "calculate_absolute_error",
    "calculate_percentage_error",
    "BenchmarkTarget",
    "ObjectAccuracyResult",
    "AccuracySummary",
    "STANDARD_BENCHMARK_TARGETS",
    "generate_planar_test_image",
    "run_accuracy_benchmark",
    "compute_accuracy_summary",
    "export_benchmark_csv",
]

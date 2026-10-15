"""Desktop graphical user interface for VisionMeasure.

Provides a clean, intuitive Tkinter interface for image loading, webcam capture,
reference-based calibration, Canny edge tuning, contour measurement, visual preview,
structured results display, and data export.
"""
from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from app.annotation import AnnotationConfig, annotate_measurements
from app.calibration import CalibrationData, calibrate
from app.contour_detection import detect_contours
from app.edge_detection import detect_edges
from app.export_results import export_csv, save_annotated_image
from app.image_input import CameraError, capture_frame, load_image
from app.measurement import measure_objects
from app.preprocessing import preprocess_image
from models.measurement_result import MeasurementResult


class VisionMeasureGUI:
    """Tkinter-based desktop interface for the VisionMeasure application."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the VisionMeasure GUI.

        Args:
            root: Root Tk window instance.
        """
        self.root = root
        self.root.title("VisionMeasure — Smart Object Dimension Measurement")
        self.root.geometry("1180x820")
        self.root.minsize(960, 680)

        # Application State
        self.current_image: Optional[np.ndarray] = None
        self.current_edges: Optional[np.ndarray] = None
        self.annotated_image: Optional[np.ndarray] = None
        self.source_image_name: str = ""
        self.calibration: Optional[CalibrationData] = None
        self.measurements: list[MeasurementResult] = []
        self.pick_points_mode: bool = False
        self.picked_points: list[Tuple[int, int]] = []
        self._photo_cache: Optional[ImageTk.PhotoImage] = None

        self._configure_styles()
        self._build_ui()
        self.set_status("Ready. Load an image or capture a webcam frame to begin.")

    def _configure_styles(self) -> None:
        """Configure ttk styles with a clean modern aesthetic."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.color_bg = "#F8FAFC"
        self.color_surface = "#FFFFFF"
        self.color_text = "#0F172A"
        self.color_muted = "#64748B"
        self.color_border = "#CBD5E1"
        self.color_primary = "#2563EB"
        self.color_success = "#15803D"
        self.color_warning = "#B45309"
        self.color_danger = "#B91C1C"

        self.root.configure(bg=self.color_bg)

        style.configure("TFrame", background=self.color_bg)
        style.configure("Surface.TFrame", background=self.color_surface, relief="flat")
        style.configure("TLabel", background=self.color_bg, foreground=self.color_text, font=("Segoe UI", 9))
        style.configure("Surface.TLabel", background=self.color_surface, foreground=self.color_text, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=self.color_bg, foreground=self.color_text, font=("Segoe UI", 14, "bold"))
        style.configure("Section.TLabel", background=self.color_surface, foreground=self.color_text, font=("Segoe UI", 10, "bold"))
        style.configure("Muted.TLabel", background=self.color_surface, foreground=self.color_muted, font=("Segoe UI", 8))
        style.configure("Status.TLabel", background=self.color_bg, foreground=self.color_text, font=("Segoe UI", 9))

        style.configure("TButton", font=("Segoe UI", 9), padding=(8, 4))
        style.configure("Primary.TButton", font=("Segoe UI", 9, "bold"), padding=(10, 5))
        style.map("Primary.TButton", background=[("active", "#1D4ED8"), ("!disabled", "#2563EB")], foreground=[("!disabled", "#FFFFFF")])

        style.configure("TLabelframe", background=self.color_surface, relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=self.color_surface, foreground=self.color_text, font=("Segoe UI", 9, "bold"))

        style.configure("Treeview", font=("Segoe UI", 9), rowheight=24)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_ui(self) -> None:
        """Construct the full application layout."""
        # Main layout container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header Title Bar
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 8))

        title_lbl = ttk.Label(header_frame, text="◈ VisionMeasure", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = ttk.Label(
            header_frame,
            text="Planar Object Dimension Measurement & Scale Calibration",
            font=("Segoe UI", 9, "italic"),
            foreground=self.color_muted,
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=(12, 0), pady=(4, 0))

        # Content Split: Left Controls Panel (scrollable if needed) and Right View Panel
        content_pane = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        content_pane.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(content_pane, width=340, padding=(0, 0, 10, 0))
        right_panel = ttk.Frame(content_pane, padding=(0, 0, 0, 0))

        content_pane.add(left_panel, weight=0)
        content_pane.add(right_panel, weight=1)

        self._build_controls_panel(left_panel)
        self._build_display_panel(right_panel)

        # Status Bar at bottom
        status_frame = ttk.Frame(main_container, relief="sunken", padding=(4, 2))
        status_frame.pack(fill=tk.X, pady=(6, 0))

        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.cal_badge_var = tk.StringVar(value="Scale: Uncalibrated")
        self.cal_badge = tk.Label(
            status_frame,
            textvariable=self.cal_badge_var,
            font=("Segoe UI", 8, "bold"),
            bg="#FEF3C7",
            fg="#92400E",
            padx=8,
            pady=2,
        )
        self.cal_badge.pack(side=tk.RIGHT)

    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        """Build the left controls panel with input, calibration, settings, and export actions."""
        # 1. Image Input Section
        input_group = ttk.LabelFrame(parent, text=" 1. Image Source ", padding="8")
        input_group.pack(fill=tk.X, pady=(0, 8))

        btn_box = ttk.Frame(input_group)
        btn_box.pack(fill=tk.X)

        self.upload_btn = ttk.Button(
            btn_box,
            text="📁 Upload Image",
            command=self.load_image_dialog,
        )
        self.upload_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.webcam_btn = ttk.Button(
            btn_box,
            text="📷 Webcam Capture",
            command=self.capture_webcam_dialog,
        )
        self.webcam_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        self.file_info_var = tk.StringVar(value="No image loaded.")
        file_info_lbl = ttk.Label(
            input_group,
            textvariable=self.file_info_var,
            font=("Segoe UI", 8),
            foreground=self.color_muted,
            wraplength=310,
        )
        file_info_lbl.pack(fill=tk.X, pady=(6, 0))

        # 2. Calibration Section
        cal_group = ttk.LabelFrame(parent, text=" 2. Reference Calibration ", padding="8")
        cal_group.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(cal_group)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Known Length:").pack(side=tk.LEFT)
        self.ref_length_var = tk.StringVar(value="5.0")
        self.ref_length_entry = ttk.Entry(row1, textvariable=self.ref_length_var, width=8)
        self.ref_length_entry.pack(side=tk.RIGHT)

        row2 = ttk.Frame(cal_group)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Unit:").pack(side=tk.LEFT)
        self.unit_var = tk.StringVar(value="cm")
        self.unit_combo = ttk.Combobox(
            row2,
            textvariable=self.unit_var,
            values=["cm", "mm"],
            state="readonly",
            width=6,
        )
        self.unit_combo.pack(side=tk.RIGHT)

        row3 = ttk.Frame(cal_group)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Pixel Span:").pack(side=tk.LEFT)
        self.ref_px_var = tk.StringVar(value="")
        self.ref_px_entry = ttk.Entry(row3, textvariable=self.ref_px_var, width=8)
        self.ref_px_entry.pack(side=tk.RIGHT)

        cal_tools_box = ttk.Frame(cal_group)
        cal_tools_box.pack(fill=tk.X, pady=(4, 2))

        self.pick_points_btn = ttk.Button(
            cal_tools_box,
            text="🎯 Click 2 Points on Preview",
            command=self.toggle_point_picker,
        )
        self.pick_points_btn.pack(fill=tk.X)

        cal_action_box = ttk.Frame(cal_group)
        cal_action_box.pack(fill=tk.X, pady=(4, 0))

        self.calibrate_btn = ttk.Button(
            cal_action_box,
            text="✓ Calibrate Scale",
            command=self.run_calibration,
        )
        self.calibrate_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.clear_cal_btn = ttk.Button(
            cal_action_box,
            text="Clear Cal",
            command=self.clear_calibration,
            width=9,
        )
        self.clear_cal_btn.pack(side=tk.RIGHT, padx=(2, 0))

        # 3. Detection Parameters (Canny & Area)
        settings_group = ttk.LabelFrame(parent, text=" 3. Detection Parameters ", padding="8")
        settings_group.pack(fill=tk.X, pady=(0, 8))

        s_row1 = ttk.Frame(settings_group)
        s_row1.pack(fill=tk.X, pady=2)
        ttk.Label(s_row1, text="Canny Low:").pack(side=tk.LEFT)
        self.canny_low_var = tk.IntVar(value=50)
        self.canny_low_spin = ttk.Spinbox(s_row1, from_=0, to_=255, textvariable=self.canny_low_var, width=6)
        self.canny_low_spin.pack(side=tk.RIGHT)

        s_row2 = ttk.Frame(settings_group)
        s_row2.pack(fill=tk.X, pady=2)
        ttk.Label(s_row2, text="Canny High:").pack(side=tk.LEFT)
        self.canny_high_var = tk.IntVar(value=150)
        self.canny_high_spin = ttk.Spinbox(s_row2, from_=1, to_=300, textvariable=self.canny_high_var, width=6)
        self.canny_high_spin.pack(side=tk.RIGHT)

        s_row3 = ttk.Frame(settings_group)
        s_row3.pack(fill=tk.X, pady=2)
        ttk.Label(s_row3, text="Min Area (px²):").pack(side=tk.LEFT)
        self.min_area_var = tk.DoubleVar(value=500.0)
        self.min_area_spin = ttk.Spinbox(s_row3, from_=10.0, to_=50000.0, increment=50.0, textvariable=self.min_area_var, width=6)
        self.min_area_spin.pack(side=tk.RIGHT)

        # 4. Primary Actions
        action_group = ttk.LabelFrame(parent, text=" 4. Execute & Export ", padding="8")
        action_group.pack(fill=tk.X, pady=(0, 8))

        self.measure_btn = ttk.Button(
            action_group,
            text="▶ Measure Objects",
            style="Primary.TButton",
            command=self.run_measurement,
        )
        self.measure_btn.pack(fill=tk.X, pady=(0, 4))

        exp_box = ttk.Frame(action_group)
        exp_box.pack(fill=tk.X, pady=2)

        self.save_img_btn = ttk.Button(
            exp_box,
            text="💾 Save Image",
            command=self.save_image_dialog,
        )
        self.save_img_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.export_csv_btn = ttk.Button(
            exp_box,
            text="📊 Export CSV",
            command=self.export_csv_dialog,
        )
        self.export_csv_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        self.reset_btn = ttk.Button(
            action_group,
            text="↺ Reset All",
            command=self.reset_all,
        )
        self.reset_btn.pack(fill=tk.X, pady=(4, 0))

    def _build_display_panel(self, parent: ttk.Frame) -> None:
        """Build the right side containing preview tabs, image canvas, and measurement results table."""
        # Top toolbar for preview mode selection
        preview_bar = ttk.Frame(parent)
        preview_bar.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(preview_bar, text="View:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))

        self.view_mode_var = tk.StringVar(value="annotated")
        modes = [("Annotated Image", "annotated"), ("Source Image", "source"), ("Edge Map", "edges")]
        for label, val in modes:
            rb = ttk.Radiobutton(
                preview_bar,
                text=label,
                value=val,
                variable=self.view_mode_var,
                command=self.update_image_display,
            )
            rb.pack(side=tk.LEFT, padx=6)

        # Canvas Frame for rendering the selected image
        self.canvas_container = ttk.Frame(parent, relief="solid", borderwidth=1)
        self.canvas_container.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.canvas = tk.Canvas(self.canvas_container, bg="#1E293B", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.update_image_display())
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Results Table Panel
        results_group = ttk.LabelFrame(parent, text=" Measurement Results ", padding="6")
        results_group.pack(fill=tk.X, pady=(0, 0))

        columns = ("id", "length", "width", "area", "unit", "status")
        self.tree = ttk.Treeview(results_group, columns=columns, show="headings", height=5)
        self.tree.heading("id", text="Object ID")
        self.tree.heading("length", text="Length")
        self.tree.heading("width", text="Width")
        self.tree.heading("area", text="Contour Area (px²)")
        self.tree.heading("unit", text="Unit")
        self.tree.heading("status", text="Calibration Status")

        self.tree.column("id", width=80, anchor=tk.CENTER)
        self.tree.column("length", width=120, anchor=tk.E)
        self.tree.column("width", width=120, anchor=tk.E)
        self.tree.column("area", width=140, anchor=tk.E)
        self.tree.column("unit", width=80, anchor=tk.CENTER)
        self.tree.column("status", width=160, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(results_group, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Planar constraint disclaimer
        disclaimer_lbl = ttk.Label(
            parent,
            text="* Note: Physical dimensions assume target objects and calibration reference are coplanar and perpendicular to camera axis.",
            font=("Segoe UI", 8, "italic"),
            foreground=self.color_muted,
        )
        disclaimer_lbl.pack(fill=tk.X, pady=(4, 0))

    # --- Image Loading & Webcam Handlers ---

    def load_image_dialog(self) -> None:
        """Open file picker to select and load an image."""
        filetypes = [
            ("Image files", "*.jpg;*.jpeg;*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*"),
        ]
        chosen_path = filedialog.askopenfilename(title="Select Target Image", filetypes=filetypes)
        if not chosen_path:
            return

        try:
            self.load_image_from_path(chosen_path)
            self.set_status(f"Loaded image '{Path(chosen_path).name}' ({self.current_image.shape[1]}x{self.current_image.shape[0]} px).")
        except Exception as exc:
            self.show_error("Image Load Error", str(exc))

    def load_image_from_path(self, path: str | Path) -> None:
        """Load image array and initialize canvas view.

        Args:
            path: String or Path to image.
        """
        img = load_image(path)
        self.current_image = img
        self.source_image_name = Path(path).name
        self.current_edges = None
        self.annotated_image = None
        self.measurements.clear()
        self._clear_results_table()
        self.file_info_var.set(f"Loaded: {self.source_image_name} ({img.shape[1]}x{img.shape[0]} px)")
        self.view_mode_var.set("source")
        self.update_image_display()

    def capture_webcam_dialog(self) -> None:
        """Capture a single frame from the connected webcam."""
        try:
            frame = capture_frame(camera_index=0)
            self.current_image = frame
            self.source_image_name = "webcam_capture.png"
            self.current_edges = None
            self.annotated_image = None
            self.measurements.clear()
            self._clear_results_table()
            self.file_info_var.set(f"Webcam Frame Captured ({frame.shape[1]}x{frame.shape[0]} px)")
            self.view_mode_var.set("source")
            self.update_image_display()
            self.set_status("Webcam frame captured successfully.")
        except CameraError as exc:
            self.show_error("Webcam Error", f"Webcam capture failed:\n{exc}")
        except Exception as exc:
            self.show_error("Capture Error", f"Unexpected error during capture:\n{exc}")

    # --- Calibration Handlers ---

    def toggle_point_picker(self) -> None:
        """Enable or disable interactive 2-point reference span picker on canvas."""
        if self.current_image is None:
            self.show_error("No Image", "Load an image first before picking calibration reference points.")
            return

        self.pick_points_mode = not self.pick_points_mode
        self.picked_points.clear()
        if self.pick_points_mode:
            self.pick_points_btn.configure(text="🎯 [Click 2 Points on Image...]")
            self.set_status("Click first reference point on the preview canvas...")
            self.view_mode_var.set("source")
            self.update_image_display()
        else:
            self.pick_points_btn.configure(text="🎯 Click 2 Points on Preview")
            self.set_status("Point picking cancelled.")
            self.update_image_display()

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Handle canvas clicks for interactive point-to-point pixel distance measurement."""
        if not self.pick_points_mode or self.current_image is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = self.current_image.shape[:2]

        scale = min(canvas_w / img_w, canvas_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        offset_x = (canvas_w - scaled_w) // 2
        offset_y = (canvas_h - scaled_h) // 2

        # Convert canvas click to original image pixel coordinates
        img_x = (event.x - offset_x) / scale
        img_y = (event.y - offset_y) / scale

        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            self.picked_points.append((int(round(img_x)), int(round(img_y))))

            if len(self.picked_points) == 1:
                self.set_status(f"Point 1 set at ({self.picked_points[0][0]}, {self.picked_points[0][1]}). Click point 2...")
                self._draw_picker_overlay()
            elif len(self.picked_points) == 2:
                p1, p2 = self.picked_points
                pixel_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                self.ref_px_var.set(f"{pixel_dist:.1f}")
                self.pick_points_mode = False
                self.pick_points_btn.configure(text="🎯 Click 2 Points on Preview")
                self.set_status(f"Reference distance measured: {pixel_dist:.1f} px between points.")
                self.picked_points.clear()
                self.update_image_display()

    def _draw_picker_overlay(self) -> None:
        """Draw interactive crosshairs on canvas for selected reference points."""
        if not self.picked_points or self.current_image is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = self.current_image.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        offset_x = (canvas_w - int(img_w * scale)) // 2
        offset_y = (canvas_h - int(img_h * scale)) // 2

        for pt in self.picked_points:
            cx = offset_x + int(pt[0] * scale)
            cy = offset_y + int(pt[1] * scale)
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#EF4444", outline="#FFFFFF", width=2)

    def run_calibration(self) -> None:
        """Validate reference length and pixel span, compute scale, and update status."""
        try:
            px_val = float(self.ref_px_var.get().strip())
        except ValueError:
            self.show_error("Calibration Error", "Please provide a valid numeric Reference Pixel Span.")
            return

        try:
            len_val = float(self.ref_length_var.get().strip())
        except ValueError:
            self.show_error("Calibration Error", "Please provide a valid numeric Known Reference Length.")
            return

        unit_val = self.unit_var.get().strip()

        try:
            cal = calibrate(reference_px=px_val, reference_length=len_val, unit=unit_val)
            self.calibration = cal
            scale_str = f"{cal.pixels_per_unit:.2f} px/{cal.unit}"
            self.cal_badge_var.set(f"Calibrated: {scale_str}")
            self.cal_badge.configure(bg="#DCFCE7", fg="#166534")
            self.set_status(f"System calibrated successfully! Scale: {scale_str}")

            # If measurements already exist, re-calculate them with the new calibration
            if self.measurements:
                self.run_measurement()

        except Exception as exc:
            self.show_error("Calibration Failed", str(exc))

    def clear_calibration(self) -> None:
        """Clear calibration data and revert to uncalibrated mode."""
        self.calibration = None
        self.cal_badge_var.set("Scale: Uncalibrated")
        self.cal_badge.configure(bg="#FEF3C7", fg="#92400E")
        self.set_status("Calibration cleared. Measurements will be displayed in pixels.")
        if self.measurements:
            self.run_measurement()

    # --- Measurement Pipeline ---

    def run_measurement(self) -> None:
        """Execute the full computer-vision measurement pipeline on the current image."""
        if self.current_image is None:
            self.show_error("No Image Loaded", "Please upload an image or capture a webcam frame first.")
            return

        try:
            canny_low = int(self.canny_low_var.get())
            canny_high = int(self.canny_high_var.get())
            min_area = float(self.min_area_var.get())

            # 1. Preprocessing (Grayscale + Gaussian Blur)
            preprocessed = preprocess_image(self.current_image, blur_kernel=5)

            # 2. Edge Detection (Canny)
            self.current_edges = detect_edges(
                preprocessed,
                threshold1=canny_low,
                threshold2=canny_high,
                apply_closing=True,
            )

            # 3. Contour Detection & Geometry
            geometries = detect_contours(self.current_edges, min_area=min_area)

            # 4. Dimension Measurement (Calibrated or Pixel-Only)
            preferred_unit = self.unit_var.get() if self.calibration else "cm"
            self.measurements = measure_objects(
                geometries=geometries,
                calibration=self.calibration,
                start_id=1,
            )

            # 5. Annotation
            cfg = AnnotationConfig(preferred_unit=preferred_unit)
            self.annotated_image = annotate_measurements(
                image=self.current_image,
                measurements=self.measurements,
                geometries=geometries,
                config=cfg,
                preferred_unit=preferred_unit,
            )

            # Update results table and display
            self._populate_results_table(self.measurements)
            self.view_mode_var.set("annotated")
            self.update_image_display()

            obj_count = len(self.measurements)
            if obj_count == 0:
                self.set_status("Measurement complete: No candidate objects found matching criteria.")
            else:
                cal_status = "calibrated physical units" if self.calibration else "pixels (uncalibrated)"
                self.set_status(f"Measurement complete: Detected {obj_count} object(s) in {cal_status}.")

        except Exception as exc:
            self.show_error("Measurement Error", f"An error occurred during measurement:\n{exc}")

    def _clear_results_table(self) -> None:
        """Remove all rows from the Treeview."""
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _populate_results_table(self, results: Sequence[MeasurementResult]) -> None:
        """Populate the measurement results Treeview.

        Args:
            results: Sequence of MeasurementResult instances.
        """
        self._clear_results_table()
        target_unit = self.unit_var.get() if self.calibration else "px"

        for res in results:
            if res.is_calibrated and self.calibration:
                if target_unit == "mm":
                    l_str = f"{res.length_mm:.1f}" if res.length_mm is not None else ""
                    w_str = f"{res.width_mm:.1f}" if res.width_mm is not None else ""
                else:
                    l_str = f"{res.length_cm:.2f}" if res.length_cm is not None else ""
                    w_str = f"{res.width_cm:.2f}" if res.width_cm is not None else ""
                unit_str = target_unit
                status_str = f"Calibrated ({res.pixels_per_cm:.1f} px/cm)"
            else:
                l_str = f"{res.length_px:.1f}"
                w_str = f"{res.width_px:.1f}"
                unit_str = "px"
                status_str = "Uncalibrated"

            area_str = f"{res.contour_area_px2:.1f}"

            self.tree.insert(
                "",
                tk.END,
                values=(res.object_id, l_str, w_str, area_str, unit_str, status_str),
            )

    # --- Image Rendering on Tk Canvas ---

    def update_image_display(self) -> None:
        """Render the selected view (source, edges, or annotated) onto the Tk Canvas."""
        mode = self.view_mode_var.get()

        target_img: Optional[np.ndarray] = None
        if mode == "annotated":
            target_img = self.annotated_image if self.annotated_image is not None else self.current_image
        elif mode == "edges":
            target_img = self.current_edges
        else:
            target_img = self.current_image

        self.canvas.delete("all")

        if target_img is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="No image available for current view",
                fill="#94A3B8",
                font=("Segoe UI", 11),
            )
            return

        canvas_w = max(self.canvas.winfo_width(), 10)
        canvas_h = max(self.canvas.winfo_height(), 10)
        img_h, img_w = target_img.shape[:2]

        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        # Convert NumPy array to RGB PIL Image
        if target_img.ndim == 2:
            # Grayscale / Binary
            pil_img = Image.fromarray(target_img).convert("RGB")
        else:
            # OpenCV BGR -> RGB
            rgb_arr = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_arr)

        resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        self._photo_cache = ImageTk.PhotoImage(resized)

        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2

        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self._photo_cache)

        if self.pick_points_mode and self.picked_points:
            self._draw_picker_overlay()

    # --- Export & Save Handlers ---

    def save_image_dialog(self) -> None:
        """Prompt user and save the current annotated image."""
        img_to_save = self.annotated_image if self.annotated_image is not None else self.current_image
        if img_to_save is None:
            self.show_error("No Image", "There is no image to save.")
            return

        filetypes = [("PNG image", "*.png"), ("JPEG image", "*.jpg;*.jpeg")]
        initial_name = f"annotated_{self.source_image_name}" if self.source_image_name else "annotated_measurement.png"
        path = filedialog.asksaveasfilename(
            title="Save Annotated Image",
            defaultextension=".png",
            initialfile=initial_name,
            filetypes=filetypes,
        )
        if not path:
            return

        try:
            saved_path = save_annotated_image(img_to_save, path, overwrite=True)
            self.set_status(f"Annotated image saved: {Path(saved_path).name}")
            messagebox.showinfo("Image Saved", f"Successfully saved image to:\n{saved_path}")
        except Exception as exc:
            self.show_error("Save Error", str(exc))

    def export_csv_dialog(self) -> None:
        """Prompt user and export current measurement records to CSV."""
        if not self.measurements:
            self.show_error("No Results", "No measurements available to export. Run measurement first.")
            return

        filetypes = [("CSV file", "*.csv")]
        initial_name = f"measurements_{Path(self.source_image_name).stem if self.source_image_name else 'data'}.csv"
        path = filedialog.asksaveasfilename(
            title="Export Measurements to CSV",
            defaultextension=".csv",
            initialfile=initial_name,
            filetypes=filetypes,
        )
        if not path:
            return

        try:
            saved_path = export_csv(
                measurements=self.measurements,
                output_path=path,
                source_image=self.source_image_name,
                overwrite=True,
            )
            self.set_status(f"Exported {len(self.measurements)} results to {Path(saved_path).name}")
            messagebox.showinfo("Export Successful", f"Saved {len(self.measurements)} records to:\n{saved_path}")
        except Exception as exc:
            self.show_error("Export Error", str(exc))

    # --- Reset & Utility Methods ---

    def reset_all(self) -> None:
        """Reset all image data, results, calibration, and UI controls consistently."""
        self.current_image = None
        self.current_edges = None
        self.annotated_image = None
        self.source_image_name = ""
        self.calibration = None
        self.measurements.clear()
        self.pick_points_mode = False
        self.picked_points.clear()
        self._photo_cache = None

        self.file_info_var.set("No image loaded.")
        self.ref_px_var.set("")
        self.ref_length_var.set("5.0")
        self.unit_var.set("cm")
        self.cal_badge_var.set("Scale: Uncalibrated")
        self.cal_badge.configure(bg="#FEF3C7", fg="#92400E")
        self.pick_points_btn.configure(text="🎯 Click 2 Points on Preview")
        self.view_mode_var.set("annotated")

        self._clear_results_table()
        self.canvas.delete("all")
        self.set_status("Application reset. Load an image to start.")

    def set_status(self, message: str, is_error: bool = False) -> None:
        """Update status bar text.

        Args:
            message: Status message to display.
            is_error: Whether the status represents an error state.
        """
        self.status_var.set(message)
        if is_error:
            self.status_label.configure(foreground=self.color_danger)
        else:
            self.status_label.configure(foreground=self.color_text)

    def show_error(self, title: str, message: str) -> None:
        """Show an error modal and log to status bar.

        Args:
            title: Dialog title.
            message: Informative error message.
        """
        self.set_status(f"Error: {message.splitlines()[0]}", is_error=True)
        messagebox.showerror(title, message)


def run_gui() -> None:
    """Launch the VisionMeasure Desktop GUI application."""
    root = tk.Tk()
    VisionMeasureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()

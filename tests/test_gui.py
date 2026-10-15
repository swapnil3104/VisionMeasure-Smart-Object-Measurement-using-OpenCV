"""Unit tests for the VisionMeasure desktop GUI application."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.gui import VisionMeasureGUI
from app.image_input import CameraError


@pytest.fixture(scope="session")
def session_tk_root() -> tk.Tk:
    """Create a single hidden Tk root window for the test session."""
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def tk_root(session_tk_root: tk.Tk) -> tk.Toplevel:
    """Create a fresh hidden Toplevel window for each test."""
    top = tk.Toplevel(session_tk_root)
    top.withdraw()
    yield top
    try:
        top.destroy()
    except Exception:
        pass


@pytest.fixture
def test_image_file(tmp_path: Path) -> Path:
    """Create a synthetic test image with a known white rectangle on black background."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # 100x60 rectangle centered
    cv2.rectangle(img, (100, 120), (200, 180), (255, 255, 255), -1)
    file_path = tmp_path / "test_gui_rect.png"
    cv2.imwrite(str(file_path), img)
    return file_path


def test_gui_initial_state(tk_root: tk.Tk) -> None:
    """Verify GUI initializes cleanly with correct default states and labels."""
    app = VisionMeasureGUI(tk_root)

    assert app.current_image is None
    assert app.calibration is None
    assert app.measurements == []
    assert "Uncalibrated" in app.cal_badge_var.get()
    assert "Ready" in app.status_var.get()
    assert len(app.tree.get_children()) == 0
    assert app.upload_btn is not None
    assert app.webcam_btn is not None
    assert app.calibrate_btn is not None
    assert app.measure_btn is not None
    assert app.reset_btn is not None


def test_gui_load_image(tk_root: tk.Tk, test_image_file: Path) -> None:
    """Verify loading an image file updates GUI state and canvas."""
    app = VisionMeasureGUI(tk_root)
    app.load_image_from_path(test_image_file)

    assert app.current_image is not None
    assert app.current_image.shape == (300, 300, 3)
    assert app.source_image_name == "test_gui_rect.png"
    assert "Loaded: test_gui_rect.png" in app.file_info_var.get()
    assert app.view_mode_var.get() == "source"


def test_gui_webcam_failure_handling(tk_root: tk.Tk, test_image_file: Path) -> None:
    """Verify webcam capture error does not crash the UI and image upload remains usable."""
    app = VisionMeasureGUI(tk_root)

    with patch("app.gui.capture_frame", side_effect=CameraError("Cannot open camera 0")):
        with patch("app.gui.messagebox.showerror") as mock_err:
            app.capture_webcam_dialog()
            assert mock_err.called
            assert "Error" in app.status_var.get()

    # Image upload remains functional after webcam error
    app.load_image_from_path(test_image_file)
    assert app.current_image is not None


def test_gui_calibration_workflow(tk_root: tk.Tk) -> None:
    """Verify calibration calculation, status badge update, and clearing."""
    app = VisionMeasureGUI(tk_root)

    # Test invalid calibration input
    app.ref_px_var.set("invalid_number")
    with patch("app.gui.messagebox.showerror") as mock_err:
        app.run_calibration()
        assert mock_err.called
        assert app.calibration is None

    # Test valid calibration input: 200 px for 5.0 cm = 40 px/cm
    app.ref_px_var.set("200.0")
    app.ref_length_var.set("5.0")
    app.unit_var.set("cm")
    app.run_calibration()

    assert app.calibration is not None
    assert pytest.approx(app.calibration.pixels_per_cm, rel=1e-3) == 40.0
    assert "Calibrated: 40.00 px/cm" in app.cal_badge_var.get()

    # Test clearing calibration
    app.clear_calibration()
    assert app.calibration is None
    assert "Uncalibrated" in app.cal_badge_var.get()


def test_gui_measurement_uncalibrated(tk_root: tk.Tk, test_image_file: Path) -> None:
    """Verify measurement without calibration displays uncalibrated pixel results."""
    app = VisionMeasureGUI(tk_root)
    app.load_image_from_path(test_image_file)

    app.min_area_var.set(200.0)
    app.run_measurement()

    assert len(app.measurements) > 0
    assert app.annotated_image is not None
    assert app.current_edges is not None

    # Verify Treeview entries
    items = app.tree.get_children()
    assert len(items) == len(app.measurements)

    row_vals = app.tree.item(items[0], "values")
    # Columns: id, length, width, area, unit, status
    assert row_vals[4] == "px"
    assert row_vals[5] == "Uncalibrated"


def test_gui_measurement_calibrated(tk_root: tk.Tk, test_image_file: Path) -> None:
    """Verify measurement with calibration displays converted physical units in table."""
    app = VisionMeasureGUI(tk_root)
    app.load_image_from_path(test_image_file)

    # Calibrate: 100 px = 2.0 cm -> 50 px/cm
    app.ref_px_var.set("100.0")
    app.ref_length_var.set("2.0")
    app.unit_var.set("cm")
    app.run_calibration()

    app.min_area_var.set(200.0)
    app.run_measurement()

    assert len(app.measurements) > 0
    first_res = app.measurements[0]
    assert first_res.is_calibrated
    assert first_res.length_cm is not None

    items = app.tree.get_children()
    row_vals = app.tree.item(items[0], "values")
    assert row_vals[4] == "cm"
    assert "Calibrated" in row_vals[5]


def test_gui_interactive_point_picker(tk_root: tk.Tk, test_image_file: Path) -> None:
    """Verify clicking 2 points on preview canvas calculates reference pixel distance."""
    app = VisionMeasureGUI(tk_root)
    app.load_image_from_path(test_image_file)

    # Mock canvas dimensions to 300x300 so scale is 1.0 and offset is 0
    app.canvas.winfo_width = MagicMock(return_value=300)
    app.canvas.winfo_height = MagicMock(return_value=300)

    app.toggle_point_picker()
    assert app.pick_points_mode is True

    # Simulate click at point 1: (50, 50)
    event1 = MagicMock()
    event1.x = 50
    event1.y = 50
    app._on_canvas_click(event1)
    assert len(app.picked_points) == 1

    # Simulate click at point 2: (50, 150) -> distance should be 100 px
    event2 = MagicMock()
    event2.x = 50
    event2.y = 150
    app._on_canvas_click(event2)

    assert app.pick_points_mode is False
    assert float(app.ref_px_var.get()) == pytest.approx(100.0, abs=0.5)


def test_gui_switch_views(tk_root: tk.Tk, test_image_file: Path) -> None:
    """Verify switching view mode between source, edges, and annotated."""
    app = VisionMeasureGUI(tk_root)
    app.load_image_from_path(test_image_file)
    app.run_measurement()

    for mode in ["source", "edges", "annotated"]:
        app.view_mode_var.set(mode)
        app.update_image_display()


def test_gui_reset_all(tk_root: tk.Tk, test_image_file: Path) -> None:
    """Verify reset_all clears current image, results, calibration, and UI state."""
    app = VisionMeasureGUI(tk_root)
    app.load_image_from_path(test_image_file)

    app.ref_px_var.set("100.0")
    app.ref_length_var.set("2.0")
    app.run_calibration()
    app.run_measurement()

    assert app.current_image is not None
    assert app.calibration is not None
    assert len(app.measurements) > 0

    app.reset_all()

    assert app.current_image is None
    assert app.calibration is None
    assert len(app.measurements) == 0
    assert "Uncalibrated" in app.cal_badge_var.get()
    assert len(app.tree.get_children()) == 0
    assert app.ref_px_var.get() == ""
    assert "reset" in app.status_var.get().lower()


def test_gui_save_and_export_dialogs(tk_root: tk.Tk, test_image_file: Path, tmp_path: Path) -> None:
    """Verify save image and export CSV dialog workflows."""
    app = VisionMeasureGUI(tk_root)
    app.load_image_from_path(test_image_file)
    app.run_measurement()

    # Save image
    out_img = tmp_path / "gui_annotated.png"
    with patch("app.gui.filedialog.asksaveasfilename", return_value=str(out_img)):
        with patch("app.gui.messagebox.showinfo") as mock_info:
            app.save_image_dialog()
            assert mock_info.called
            assert out_img.is_file()

    # Export CSV
    out_csv = tmp_path / "gui_results.csv"
    with patch("app.gui.filedialog.asksaveasfilename", return_value=str(out_csv)):
        with patch("app.gui.messagebox.showinfo") as mock_info:
            app.export_csv_dialog()
            assert mock_info.called
            assert out_csv.is_file()

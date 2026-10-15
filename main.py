"""VisionMeasure — Real-Time Object Detection & Dimension Measurement.

Application entry point supporting the modern YOLO Web Application
and legacy desktop GUI.
"""
import argparse
import sys


def main() -> int:
    """Launch the VisionMeasure application."""
    parser = argparse.ArgumentParser(
        description="VisionMeasure AI — Smart Object Detection & Dimension Measurement"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify dependencies and environment integrity",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch legacy desktop Tkinter GUI instead of Web Application",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for web server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for web server (default: 8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="YOLO model variant (e.g. yolov8m.pt, yolov8x.pt, yolov8s.pt, yolov8n.pt)",
    )

    args = parser.parse_args()

    if args.check:
        print("Checking VisionMeasure environment...")
        import cv2
        import numpy as np
        import ultralytics

        print(f"OpenCV: {cv2.__version__}, NumPy: {np.__version__}")
        print(f"Ultralytics YOLO: {ultralytics.__version__}")
        from app.yolo_detector import YOLOMeasurementEngine

        YOLOMeasurementEngine(args.model)
        print(f"VisionMeasure YOLO Engine ({args.model}) and Web modules verified successfully.")
        return 0

    if args.gui:
        from app.gui import run_gui

        print("Launching legacy VisionMeasure desktop Tkinter GUI...")
        run_gui()
        return 0

    # Default: Modern Web Application
    from app.web_server import get_yolo_engine, run_server

    get_yolo_engine(args.model)
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

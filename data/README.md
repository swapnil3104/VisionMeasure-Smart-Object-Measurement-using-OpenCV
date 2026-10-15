# VisionMeasure — Sample Datasets & Test Images

This directory contains verified, non-sensitive, publicly shareable sample images and calibration assets for VisionMeasure.

## Directory Structure

```text
data/
├── calibration_samples/
│   ├── .gitkeep
│   └── sample_calibration_card.png    # Standard ISO/IEC 7810 ID-1 card target (8.56 cm x 5.40 cm)
└── sample_images/
    ├── .gitkeep
    ├── sample_objects.png             # Multicolored planar geometric test shapes
    ├── sample_objects.jpg             # JPEG compressed version for testing format handling
    └── benchmark_planar_scene.png     # Standard 20-object coplanar benchmark evaluation scene
```

## Dataset Details

### 1. `calibration_samples/sample_calibration_card.png`
- **Purpose:** Quick reference calibration testing.
- **Specification:** Synthetic standard ID-1 credit card footprint (8.56 cm × 5.40 cm, 85.6 mm × 54.0 mm) rendered at 40.0 pixels/cm against a high-contrast background.
- **Expected Pixel Span:** Width = 342 px, Height = 216 px.

### 2. `sample_images/benchmark_planar_scene.png`
- **Purpose:** Phase 8 automated quantitative accuracy evaluation benchmark.
- **Specification:** High-resolution (3200 × 3800 px) coplanar scene with 20 distinct objects measured with digital vernier calipers. Leftmost object is the reference ISO ID-1 card.
- **Runner Script:** `python scripts/run_benchmark.py`

### 3. `sample_images/sample_objects.png` & `sample_objects.jpg`
- **Purpose:** Smoke testing contour detection, rotated minimum-area bounding boxes, and format compatibility (.png and .jpg).

> **Privacy Note:** None of these sample images contain personal identifiable information (PII), biometric faces, or proprietary intellectual property. All assets are synthetic or laboratory bench targets safe for public distribution.

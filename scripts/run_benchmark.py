"""CLI runner to execute Phase 8 Accuracy Evaluation Benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.accuracy_benchmark import (  # noqa: E402
    STANDARD_BENCHMARK_TARGETS,
    run_accuracy_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VisionMeasure Phase 8 - Physical Accuracy Benchmark Evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save the CSV report and annotated verification image",
    )
    parser.add_argument(
        "--pixels-per-cm",
        type=float,
        default=40.0,
        help="Scale factor (pixels/cm) for the synthetic coplanar test scene",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print("=" * 70)
    print("VisionMeasure — Phase 8 Accuracy Evaluation Benchmark")
    print("=" * 70)
    print(f"Total Benchmark Targets: {len(STANDARD_BENCHMARK_TARGETS)}")
    print(f"Reference Scale:         {args.pixels_per_cm:.1f} px/cm")
    print(f"Output Directory:        {out_dir.resolve()}")
    print("-" * 70)

    try:
        results, summary = run_accuracy_benchmark(
            targets=STANDARD_BENCHMARK_TARGETS,
            pixels_per_cm=args.pixels_per_cm,
            reference_target_index=0,  # ISO Credit Card ID-1
            output_dir=out_dir,
        )
    except Exception as exc:
        print(f"Error during accuracy benchmark execution: {exc}", file=sys.stderr)
        return 1

    print("\nRESULTS TABLE (Length & Width):")
    print(f"{'Target Name':<28} | {'Act L':>6} | {'Est L':>6} | {'L Err(%)':>8} | {'Act W':>6} | {'Est W':>6} | {'W Err(%)':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.target_name:<28} | {r.actual_length_cm:6.2f} | {r.estimated_length_cm:6.2f} | "
            f"{r.length_pct_error:7.2f}% | {r.actual_width_cm:6.2f} | {r.estimated_width_cm:6.2f} | "
            f"{r.width_pct_error:7.2f}%"
        )

    print("-" * 80)
    print("\nBENCHMARK STATISTICAL SUMMARY:")
    print(f"  * Total Targets Evaluated:         {summary.total_objects}")
    print(f"  * Mean Length Absolute Error:      {summary.mean_length_abs_error_cm:.4f} cm ({summary.mean_length_abs_error_cm * 10:.3f} mm)")
    print(f"  * Mean Length Percentage Error:    {summary.mean_length_pct_error:.2f}%")
    print(f"  * Mean Width Absolute Error:       {summary.mean_width_abs_error_cm:.4f} cm ({summary.mean_width_abs_error_cm * 10:.3f} mm)")
    print(f"  * Mean Width Percentage Error:     {summary.mean_width_pct_error:.2f}%")
    print(f"  * Max Length Percentage Error:     {summary.max_length_pct_error:.2f}%")
    print(f"  * Max Width Percentage Error:      {summary.max_width_pct_error:.2f}%")
    print(f"  * Overall Mean Absolute Error:     {summary.overall_mean_abs_error_cm:.4f} cm ({summary.overall_mean_abs_error_cm * 10:.3f} mm)")
    print(f"  * Overall Mean Percentage Error:   {summary.overall_mean_pct_error:.2f}%")
    print("=" * 70)
    print(f"[OK] Reports successfully exported to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

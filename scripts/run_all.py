"""Non-interactive V11 entry point for AutoDL and local execution."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete V11 workflow")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-free-gpu-gb", type=float, default=2.0)
    parser.add_argument("--properties", nargs="+", default=None)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--skip-environment-check", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-reliability", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def run(script_name: str, arguments: list[str], description: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name), *arguments]
    print("\n" + "=" * 80, flush=True)
    print(f"START: {description}", flush=True)
    print("RUN:", " ".join(command), flush=True)
    started = time.time()
    subprocess.run(command, check=True, cwd=ROOT)
    print(
        f"DONE: {description} ({(time.time() - started) / 60:.1f} min)",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print("V11 repeated 5x5 cross-fitted reliability workflow")
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Root: {ROOT}")
    print("=" * 80, flush=True)

    if not args.skip_environment_check:
        run("00_check_environment.py", [], "environment check")

    selection_args: list[str] = []
    if args.properties:
        selection_args.extend(["--properties", *args.properties])
    elif args.test_mode:
        selection_args.append("--test-mode")

    resume_args = [] if args.overwrite_existing else ["--skip-existing"]
    if not args.skip_training:
        run(
            "01_train_repeated5fold_cv.py",
            [
                "--device",
                args.device,
                "--min-free-gpu-gb",
                str(args.min_free_gpu_gb),
                *selection_args,
                *resume_args,
            ],
            "strict 5-repeat x 5-fold training and prediction",
        )

    if not args.skip_reliability:
        reliability_selection = (
            ["--properties", *args.properties] if args.properties else []
        )
        run(
            "02_compute_reliability.py",
            [*reliability_selection, *resume_args],
            "fold-matched reliability calculation",
        )

    if not args.skip_plots:
        if args.test_mode or args.properties:
            print(
                "\nNOTE: plots are being generated from the currently available "
                "property outputs, which may be a partial test set.",
                flush=True,
            )
        plot_steps = [
            ("04_plot_p0a_calibration.py", "P0-A calibration"),
            ("05_plot_p0b_risk_coverage.py", "P0-B risk coverage"),
            ("06_plot_p0c_structural.py", "P0-C structural support"),
            ("07_plot_p0d_extrapolation.py", "P0-D distance-gradient validation"),
            ("09_plot_p1a_property_gap.py", "P1-A property gaps"),
            ("10_plot_p1b_chemical_map.py", "P1-B chemical map"),
            ("11_plot_p1c_cluster_heatmap.py", "P1-C cluster heatmap"),
            ("12_plot_p1d_top_targets.py", "P1-D target list"),
        ]
        for script_name, description in plot_steps:
            run(script_name, [], description)

    print("\n" + "=" * 80)
    print(f"V11 complete: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Outputs: {ROOT / 'outputs'}")
    print("=" * 80, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

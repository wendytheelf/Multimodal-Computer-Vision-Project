#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
=============================================================================
04 — CNN baseline (notebook-style script)
=============================================================================

Sections:
  * Run CNN training
  * Load metrics JSON
  * Inspect predictions + timeline figures

From project root::

    python notebooks/04_cnn_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
METRICS = ROOT / "outputs" / "metrics" / "cnn_metrics.json"
PREDS = ROOT / "outputs" / "predictions" / "cnn_predictions.csv"
LABELS = ROOT / "data" / "labels" / "contact_frames.csv"
FIG_DIR = ROOT / "outputs" / "figures"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =============================================================================
# SECTION: Run CNN training
# =============================================================================
def run_training() -> None:
    section("Run CNN baseline (train_cnn.py)")
    script = SRC / "train_cnn.py"
    if not script.is_file():
        print(f"Missing {script}")
        return
    cmd = [sys.executable, str(script)]
    print("Running:", " ".join(cmd))
    print("(Requires torch, torchvision, OpenCV; GPU optional.)")
    subprocess.run(cmd, cwd=str(ROOT), check=False)


# =============================================================================
# SECTION: Load metrics
# =============================================================================
def run_load_metrics() -> None:
    section("Load outputs/metrics/cnn_metrics.json")
    if not METRICS.is_file():
        print(f"Not found: {METRICS}")
        return
    with open(METRICS, encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2))


# =============================================================================
# SECTION: View predictions + timelines
# =============================================================================
def run_predictions_and_plots() -> None:
    section("Predictions sample + figures (confusion_cnn, timelines)")
    import pandas as pd

    viz = SRC / "visualize_results.py"
    if not PREDS.is_file():
        print(f"Not found: {PREDS}")
        return
    df = pd.read_csv(PREDS)
    print(df.head(10).to_string(index=False))

    if viz.is_file():
        cmd = [
            sys.executable,
            str(viz),
            "--pred",
            str(PREDS),
            "--labels",
            str(LABELS),
            "--out-dir",
            str(FIG_DIR),
            "--name",
            "cnn",
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=False)
        print(f"Figures: {FIG_DIR}/confusion_cnn.png, timeline_cnn_*.png")


# =============================================================================
# SECTION: Compare folds (per-video holdout is implicit in pooled CSV)
# =============================================================================
def run_fold_notes() -> None:
    section("Fold notes (LOO)")

    import pandas as pd

    if not PREDS.is_file():
        return
    df = pd.read_csv(PREDS)
    print("Each row is one window; LOO trains N-1 videos and tests 1 per fold.")
    print("Pooled CSV mixes all test folds — for per-fold metrics, re-run with logging in train_cnn.")
    print("\nVideos in predictions:", sorted(df["video_name"].unique().tolist()))
    print("Rows per video:")
    print(df.groupby("video_name").size())


def main() -> None:
    run_training()
    run_load_metrics()
    run_predictions_and_plots()
    run_fold_notes()
    print("\nDone.")


if __name__ == "__main__":
    main()

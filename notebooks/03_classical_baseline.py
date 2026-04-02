#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
=============================================================================
03 — Classical baseline (notebook-style script)
=============================================================================

Sections:
  * Run classical training (subprocess)
  * Load metrics JSON
  * Confusion matrix figure (via visualize_results)
  * Inspect predictions CSV

From project root::

    python notebooks/03_classical_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
METRICS = ROOT / "outputs" / "metrics" / "classical_metrics.json"
PREDS = ROOT / "outputs" / "predictions" / "classical_predictions.csv"
LABELS = ROOT / "data" / "labels" / "contact_frames.csv"
FIG_DIR = ROOT / "outputs" / "figures"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =============================================================================
# SECTION: Run classical training
# =============================================================================
def run_training() -> None:
    section("Run classical baseline (train_classical.py)")
    script = SRC / "train_classical.py"
    if not script.is_file():
        print(f"Missing {script}")
        return
    cmd = [sys.executable, str(script)]
    print("Running:", " ".join(cmd))
    print("(Requires OpenCV, sklearn, dataset + raw videos.)")
    subprocess.run(cmd, cwd=str(ROOT), check=False)


# =============================================================================
# SECTION: Load metrics
# =============================================================================
def run_load_metrics() -> None:
    section("Load outputs/metrics/classical_metrics.json")
    if not METRICS.is_file():
        print(f"Not found: {METRICS} — run training first.")
        return
    with open(METRICS, encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))


# =============================================================================
# SECTION: Confusion matrix figure
# =============================================================================
def run_confusion_figure() -> None:
    section("Confusion matrix → outputs/figures/")
    viz = SRC / "visualize_results.py"
    if not viz.is_file() or not PREDS.is_file():
        print("Need visualize_results.py and classical_predictions.csv")
        return
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
        "classical",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    print(f"Check: {FIG_DIR}/confusion_classical.png")


# =============================================================================
# SECTION: Inspect predictions
# =============================================================================
def run_inspect_predictions() -> None:
    section("Inspect classical_predictions.csv (head + errors)")
    import pandas as pd

    if not PREDS.is_file():
        print(f"Not found: {PREDS}")
        return
    df = pd.read_csv(PREDS)
    print(df.head(10).to_string(index=False))
    wrong = df[df["true_label"] != df["pred_label"]]
    print(f"\nMisclassified windows: {len(wrong)} / {len(df)}")
    print(wrong.head(15).to_string(index=False))


def main() -> None:
    run_training()
    run_load_metrics()
    run_confusion_figure()
    run_inspect_predictions()
    print("\nDone.")


if __name__ == "__main__":
    main()

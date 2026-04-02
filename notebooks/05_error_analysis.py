#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
=============================================================================
05 — Error analysis: classical vs CNN (notebook-style script)
=============================================================================

Sections:
  * Load both prediction CSVs
  * Side-by-side frame-level agreement
  * Event-level (argmax score) errors via evaluate.py
  * Timeline plots for interesting videos
  * Scratch space for failure-pattern notes

From project root::

    python notebooks/05_error_analysis.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CLASSICAL = ROOT / "outputs" / "predictions" / "classical_predictions.csv"
CNN = ROOT / "outputs" / "predictions" / "cnn_predictions.csv"
LABELS = ROOT / "data" / "labels" / "contact_frames.csv"
FIG_DIR = ROOT / "outputs" / "figures"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import evaluate_predictions, load_contact_labels_csv, load_predictions_csv  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =============================================================================
# SECTION: Load predictions
# =============================================================================
def run_load_both() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    section("Load classical and CNN prediction CSVs")

    c_df = None
    n_df = None
    if CLASSICAL.is_file():
        c_df = load_predictions_csv(CLASSICAL)
        print(f"Classical rows: {len(c_df)}")
    else:
        print(f"Missing {CLASSICAL}")

    if CNN.is_file():
        n_df = load_predictions_csv(CNN)
        print(f"CNN rows: {len(n_df)}")
    else:
        print(f"Missing {CNN}")
    return c_df, n_df


# =============================================================================
# SECTION: Compare predictions (where both exist)
# =============================================================================
def run_compare_window_level(c_df: pd.DataFrame | None, n_df: pd.DataFrame | None) -> None:
    section("Window-level: when do baselines agree?")

    if c_df is None or n_df is None:
        print("Need both CSVs.")
        return

    key_cols = ["video_name", "center_frame"]
    merged = c_df.merge(
        n_df,
        on=key_cols,
        suffixes=("_clf", "_cnn"),
        how="inner",
    )
    if merged.empty:
        print("No overlapping (video_name, center_frame) keys.")
        return

    merged["both_correct"] = (merged["true_label_clf"] == merged["pred_label_clf"]) & (
        merged["true_label_cnn"] == merged["pred_label_cnn"]
    )
    merged["both_wrong"] = (merged["true_label_clf"] != merged["pred_label_clf"]) & (
        merged["true_label_cnn"] != merged["pred_label_cnn"]
    )
    merged["disagree"] = merged["pred_label_clf"] != merged["pred_label_cnn"]

    print(f"Merged windows: {len(merged)}")
    print("Agreement on pred_label:", (~merged["disagree"]).mean())
    print("Both wrong:", merged["both_wrong"].mean())
    print("\nSample disagreements:")
    dis = merged[merged["disagree"]].head(12)
    print(dis[key_cols + ["true_label_clf", "pred_label_clf", "pred_label_cnn", "score_clf", "score_cnn"]].to_string(index=False))


# =============================================================================
# SECTION: Event-level metrics (evaluate.py)
# =============================================================================
def run_event_metrics() -> None:
    section("Event-level metrics (pooled LOO predictions)")

    if not LABELS.is_file():
        print(f"Missing {LABELS}")
        return

    labels = load_contact_labels_csv(LABELS)
    for name, path in [("classical", CLASSICAL), ("cnn", CNN)]:
        if not path.is_file():
            continue
        df = load_predictions_csv(path)
        report = evaluate_predictions(df, labels)
        print(f"\n=== {name} ===")
        print("Frame-level:", json.dumps(report["frame_level"], indent=2))
        print("Event-level:", json.dumps(report["event_level"], indent=2))


# =============================================================================
# SECTION: Visualize timelines (both runs)
# =============================================================================
def run_timelines() -> None:
    section("Regenerate timeline figures for both baselines")

    viz = SRC / "visualize_results.py"
    if not viz.is_file():
        return
    for pred, tag in [(CLASSICAL, "classical"), (CNN, "cnn")]:
        if not pred.is_file():
            continue
        cmd = [
            sys.executable,
            str(viz),
            "--pred",
            str(pred),
            "--labels",
            str(LABELS),
            "--out-dir",
            str(FIG_DIR),
            "--name",
            tag,
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=False)


# =============================================================================
# SECTION: Failure patterns — write short notes for your report
# =============================================================================
def run_failure_notes() -> None:
    section("Failure patterns checklist (fill in for Check-In 2)")

    notes = """
    - Motion blur on fast swing → weak edges / HOG; CNN may still misfire.
    - Small ball / far camera → little visual evidence at contact.
    - Occlusion (net, arms) → ambiguous appearance.
    - ±1 frame label noise → hurts strict argmax comparison.
    - Class imbalance → model biased toward negatives; check recall on positives.
    - Small N in LOO → high variance across folds.

    Edit this list in notebooks/05_error_analysis.py after inspecting timelines under outputs/figures/.
    """
    print(notes)


def main() -> None:
    c_df, n_df = run_load_both()
    run_compare_window_level(c_df, n_df)
    run_event_metrics()
    run_timelines()
    run_failure_notes()
    print("\nDone.")


if __name__ == "__main__":
    main()

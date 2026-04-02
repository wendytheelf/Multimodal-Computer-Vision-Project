#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
=============================================================================
01 — Exploratory Data Analysis (notebook-style script)
=============================================================================

Volleyball contact-frame project: quick overview of labels and raw videos.

Run from the **project root**::

    python notebooks/01_eda.py

Or paste sections into an interactive session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = ROOT / "data" / "labels" / "contact_frames.csv"
RAW_DIR = ROOT / "data" / "raw_videos"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =============================================================================
# SECTION: Dataset overview
# =============================================================================
def run_dataset_overview() -> None:
    section("Dataset overview")
    print(f"Project root: {ROOT}")
    print(f"Labels CSV:   {LABELS_CSV} (exists={LABELS_CSV.is_file()})")
    print(f"Raw videos:   {RAW_DIR} (exists={RAW_DIR.is_dir()})")


# =============================================================================
# SECTION: Video count + fps / total frames summary
# =============================================================================
def run_video_and_label_summary() -> None:
    section("Video count & per-clip stats (from labels CSV)")

    if not LABELS_CSV.is_file():
        print("Create data/labels/contact_frames.csv first (see README).")
        return

    df = pd.read_csv(LABELS_CSV)
    print(f"Number of labeled videos: {len(df)}")
    print(df.head())

    if "fps" in df.columns and "total_frames" in df.columns:
        print("\nFPS summary:")
        print(df["fps"].describe())
        print("\nTotal frames summary:")
        print(df["total_frames"].describe())
        print(f"\nApprox. mean clip duration (s): {(df['total_frames'] / df['fps']).mean():.2f}")

    # Cross-check raw folder
    if RAW_DIR.is_dir():
        vids = sorted(RAW_DIR.glob("*.mp4")) + sorted(RAW_DIR.glob("*.MP4"))
        print(f"\n.mp4 files under raw_videos: {len(vids)}")
        names = {p.name for p in vids}
        missing = [n for n in df["video_name"].astype(str) if n not in names]
        if missing:
            print("Warning: label entries with no matching raw file:", missing)
        else:
            print("All labeled video_name values have a matching .mp4 in raw_videos.")


# =============================================================================
# SECTION: Label distribution (contact frame positions)
# =============================================================================
def run_label_distribution() -> None:
    section("Label distribution (where contact falls in each clip)")

    if not LABELS_CSV.is_file():
        return

    df = pd.read_csv(LABELS_CSV)
    if "contact_frame" not in df.columns or "total_frames" not in df.columns:
        print("Need contact_frame and total_frames columns.")
        return

    rel = df["contact_frame"] / df["total_frames"].replace(0, np.nan)
    print("Contact frame index (relative position in clip):")
    print(rel.describe())
    print(
        "\nInterpretation: values near 0 or 1 mean contact very early/late in the trimmed clip.\n"
        "Use this to catch annotation or trimming inconsistencies."
    )


# =============================================================================
# SECTION: Sample frames around contact (optional; requires OpenCV + files)
# =============================================================================
def run_sample_frames_around_contact(max_videos: int = 2, window: int = 2) -> None:
    section("Sample frames around annotated contact (prints frame indices)")

    try:
        import cv2
    except ImportError:
        print("Install opencv-python to decode frames (`pip install opencv-python`).")
        return

    if not LABELS_CSV.is_file() or not RAW_DIR.is_dir():
        return

    labels = pd.read_csv(LABELS_CSV)
    for _, row in labels.head(max_videos).iterrows():
        name = str(row["video_name"])
        c = int(row["contact_frame"])
        path = RAW_DIR / name
        if not path.is_file():
            print(f"Skip (missing file): {path}")
            continue
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nVideo {name}: contact_frame={c}, cap reports ~{n} frames")
        for off in range(-window, window + 1):
            idx = c + off
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            print(f"  frame {idx}: read_ok={ok}, shape={None if frame is None else frame.shape}")
        cap.release()
    print(
        "\nTip: for plots, save frames with extract_frames.py or imshow in a notebook.\n"
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    run_dataset_overview()
    run_video_and_label_summary()
    run_label_distribution()
    run_sample_frames_around_contact()
    print("\nDone.")


if __name__ == "__main__":
    main()

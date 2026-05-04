"""
Interactive contact-frame annotation for spike clips.

Writes ``data/labels/contact_frames.csv`` with columns:
  video_name, contact_frame, fps, total_frames

**Important:** This script **merges** with any existing CSV so you do not lose
labels for videos you already annotated. By default, only videos **not yet** in
the CSV are shown (``--only-missing``). Use ``--all`` to re-walk every clip.

Controls during playback:
  a / d — previous / next frame
  s     — save this frame as contact
  q     — skip this video (no label written for it this run; existing CSV row is kept)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_DIR = PROJECT_ROOT / "data" / "spike_clips"
OUTPUT_CSV = PROJECT_ROOT / "data" / "labels" / "contact_frames.csv"

WINDOW_NAME = "Contact Annotation"
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


def _list_videos(video_dir: Path) -> list[str]:
    valid_exts = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV")
    if not video_dir.is_dir():
        return []
    return sorted(
        f for f in os.listdir(video_dir) if f.lower().endswith(valid_exts)
    )


def load_existing_labels(path: Path) -> dict[str, dict[str, float | int]]:
    """``video_name -> {contact_frame, fps, total_frames}`` for rows in the CSV."""
    if not path.is_file():
        return {}
    out: dict[str, dict[str, float | int]] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("video_name"):
                continue
            name = str(row["video_name"]).strip()
            out[name] = {
                "contact_frame": int(float(row["contact_frame"])),
                "fps": float(row["fps"]),
                "total_frames": int(float(row["total_frames"])),
            }
    return out


def save_labels(
    path: Path,
    labels: dict[str, dict[str, float | int]],
    video_dir: Path,
) -> None:
    """Write CSV sorted by video name, only for videos that still exist on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, float, int]] = []
    for name in sorted(labels.keys()):
        vp = video_dir / name
        if not vp.is_file():
            continue
        d = labels[name]
        rows.append(
            (name, int(d["contact_frame"]), float(d["fps"]), int(d["total_frames"]))
        )
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "contact_frame", "fps", "total_frames"])
        w.writerows(rows)
    print(f"\nSaved {len(rows)} row(s) to: {path}")


def load_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps, total_frames


def draw_frame(frame, frame_idx, total_frames, video_name: str):
    display = frame.copy()
    display = cv2.resize(display, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    text_1 = f"Video: {video_name}"
    text_2 = f"Frame: {frame_idx}/{total_frames - 1}"
    text_3 = "a: prev  d: next  s: save  q: quit/skip"

    cv2.putText(display, text_1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, text_2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        display, text_3, (20, DISPLAY_HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    return display


def annotate_video(
    video_path: Path,
) -> tuple[int | None, float, int]:
    """
    Returns (contact_frame or None, fps, total_frames). None if user skipped
    or file unreadable.
    """
    video_name = video_path.name
    frames, fps, total_frames = load_video_frames(video_path)

    if not frames or total_frames < 1:
        print(f"Could not read frames from {video_name}")
        return None, fps, max(total_frames, 0)

    frame_idx = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, 100, 80)

    while True:
        display = draw_frame(frames[frame_idx], frame_idx, total_frames, video_name)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("d"):
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == ord("a"):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord("s"):
            print(
                f"Saved {video_name}: contact_frame={frame_idx}, "
                f"fps={fps}, total_frames={total_frames}"
            )
            cv2.destroyWindow(WINDOW_NAME)
            return frame_idx, fps, total_frames
        elif key == ord("q"):
            print(f"Skipped {video_name} (no new label for this run)")
            cv2.destroyWindow(WINDOW_NAME)
            return None, fps, total_frames

    return None, fps, total_frames  # pragma: no cover


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate ball-hand contact frame for each spike clip (GUI)."
    )
    parser.add_argument(
        "--spike-clips",
        type=Path,
        default=VIDEO_DIR,
        help="Directory containing .mp4 files (default: data/spike_clips).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV (default: data/labels/contact_frames.csv).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-annotate every video in the folder (not just ones missing from the CSV).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which videos are missing from the CSV and exit (no GUI).",
    )
    args = parser.parse_args()
    only_missing = not args.all

    video_dir = args.spike_clips
    out_path = args.output

    video_files = _list_videos(video_dir)
    if not video_files:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    labels = load_existing_labels(out_path)

    if args.dry_run:
        on_disk = set(video_files)
        labeled = set(labels.keys())
        missing = sorted(on_disk - labeled)
        extra = sorted(labeled - on_disk)
        print(f"Labels file: {out_path}")
        print(f"Videos in {video_dir}: {len(video_files)}")
        print(f"Labeled: {len(labeled)}")
        if missing:
            print(f"Missing labels (need annotation): {len(missing)}")
            for m in missing:
                print(f"  - {m}")
        else:
            print("Missing labels: none (every file in folder has a CSV row).")
        if extra:
            print(f"CSV rows with no file on disk (stale): {', '.join(extra)}")
        print("\nRun: python annotate_contact.py")
        sys.exit(0)
    n_existing = len(labels)
    to_process: list[str] = []
    for name in video_files:
        if only_missing and name in labels:
            continue
        to_process.append(name)

    print(f"Videos in folder: {len(video_files)}")
    print(f"Rows already in {out_path.name}: {n_existing}")
    if only_missing:
        print(f"To annotate (missing or use --all): {len(to_process)} -> {', '.join(to_process) or '(none)'}")
    else:
        print(f"Will re-annotate all {len(video_files)} video(s).")

    for video_name in to_process if only_missing else video_files:
        video_path = video_dir / video_name
        contact_frame, fps, total_frames = annotate_video(video_path)
        if contact_frame is not None and total_frames >= 3:
            labels[video_name] = {
                "contact_frame": int(contact_frame),
                "fps": float(fps),
                "total_frames": int(total_frames),
            }
        elif contact_frame is not None and total_frames < 3:
            print(f"[skip] {video_name} has fewer than 3 frames; not written.")

    save_labels(out_path, labels, video_dir)

    # Reminder: anything still in folder but not in CSV
    on_disk = set(_list_videos(video_dir))
    labeled = set(load_existing_labels(out_path).keys())
    still_missing = sorted(on_disk - labeled)
    if still_missing:
        print(
            "\n[note] No label row (yet) for — run again, or use --all:\n  "
            + ", ".join(still_missing)
        )
    else:
        print("\n[ok] Every video in the folder has a row in the labels CSV.")


if __name__ == "__main__":
    main()

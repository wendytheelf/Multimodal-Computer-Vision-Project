"""
Build a window-level binary classification dataset for volleyball contact-frame detection.

For each video, for every valid center frame ``t`` (where frames ``t-1, t, t+1`` exist),
we emit one row:

  - Positive (label=1) if ``t`` is within ``tolerance`` frames of the annotated contact.
  - Negative (label=0) otherwise.

Frame indices are **0-based**, consistent with OpenCV and typical CSV annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "labels" / "contact_frames.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "dataset_windows.csv"


def load_labels_csv(labels_path: str | Path) -> pd.DataFrame:
    """
    Read ``contact_frames.csv`` with columns:

      video_name, contact_frame, fps, total_frames

    Raises:
        FileNotFoundError: if path missing.
        ValueError: if required columns are absent.
    """
    path = Path(labels_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"video_name", "contact_frame", "fps", "total_frames"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns {missing}: {path}")

    # Normalize types
    df = df.copy()
    df["video_name"] = df["video_name"].astype(str)
    df["contact_frame"] = df["contact_frame"].astype(int)
    df["fps"] = df["fps"].astype(float)
    df["total_frames"] = df["total_frames"].astype(int)

    # Enforce non-negative frame counts
    if (df["total_frames"] < 3).any():
        bad = df.loc[df["total_frames"] < 3, "video_name"].tolist()
        raise ValueError(
            f"Need at least 3 frames per video to form (t-1,t,t+1) windows. Check: {bad}"
        )

    return df


def valid_center_indices(total_frames: int) -> range:
    """
    Center indices ``t`` such that ``t-1``, ``t``, ``t+1`` are all valid.

    For ``total_frames`` = N (frames indexed 0 .. N-1), valid ``t`` satisfies:
      1 <= t <= N-2

    Returns:
        range object usable in a for-loop.
    """
    if total_frames < 3:
        return range(0)  # empty
    return range(1, total_frames - 1)


def window_label(center_t: int, contact_frame: int, tolerance: int) -> int:
    """
    Binary label: 1 if |center_t - contact_frame| <= tolerance, else 0.
    """
    return 1 if abs(int(center_t) - int(contact_frame)) <= int(tolerance) else 0


def build_samples_for_video(
    video_name: str,
    contact_frame: int,
    total_frames: int,
    *,
    tolerance: int,
) -> list[dict]:
    """
    Create one dict per valid center frame for a single video.
    """
    rows: list[dict] = []
    for t in valid_center_indices(total_frames):
        rows.append(
            {
                "video_name": video_name,
                "center_frame": t,
                "prev_frame": t - 1,
                "curr_frame": t,
                "next_frame": t + 1,
                "label": window_label(t, contact_frame, tolerance),
                "contact_frame": int(contact_frame),
            }
        )
    return rows


def build_dataset_df(labels_df: pd.DataFrame, *, tolerance: int = 2) -> pd.DataFrame:
    """
    Concatenate all per-video rows into one DataFrame.

    Column order is stable for readability in spreadsheets.
    """
    all_rows: list[dict] = []
    for _, row in labels_df.iterrows():
        all_rows.extend(
            build_samples_for_video(
                str(row["video_name"]),
                int(row["contact_frame"]),
                int(row["total_frames"]),
                tolerance=tolerance,
            )
        )

    cols = [
        "video_name",
        "center_frame",
        "prev_frame",
        "curr_frame",
        "next_frame",
        "label",
        "contact_frame",
    ]
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df.reindex(columns=cols)
    return df[cols]


def save_dataset_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write dataset CSV; create parent directories if needed."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out.resolve()


# ---------------------------------------------------------------------------
# Leave-one-video-out (LOO) — split by **video_name** only (no frame leakage)
# ---------------------------------------------------------------------------


def unique_video_names(df: pd.DataFrame) -> list[str]:
    """Sorted unique video_name values in ``df``."""
    return sorted(df["video_name"].unique().tolist())


def iter_leave_one_video_out_splits(
    video_names: Iterable[str],
) -> Iterable[tuple[list[str], str]]:
    """
    Yield (train_videos, test_video) for each held-out video.

    Example:
        videos = ['a.mp4','b.mp4','c.mp4']
        -> train [b,c] test a; train [a,c] test b; train [a,b] test c
    """
    names = list(video_names)
    for test in names:
        train = [v for v in names if v != test]
        yield train, test


def split_by_test_video(df: pd.DataFrame, test_video: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a window-level DataFrame into train / test by **video_name**.

    Args:
        df: Must contain column ``video_name`` (e.g. full ``build_dataset_df`` output).
        test_video: Name exactly as in CSV (e.g. ``clip_01.mp4``).

    Returns:
        (train_df, test_df) — no overlapping videos between the two.
    """
    test_mask = df["video_name"] == test_video
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()
    return train_df, test_df


def get_loo_fold(df: pd.DataFrame, test_video: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience alias: same as ``split_by_test_video`` (LOO = pick one test video).
    """
    return split_by_test_video(df, test_video)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build window-level binary classification CSV from contact_frames.csv"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to contact_frames.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output dataset CSV path",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="Frames within ±tolerance of contact_frame get label 1 (default 2).",
    )
    parser.add_argument(
        "--print-loo",
        action="store_true",
        help="Print leave-one-video-out fold sizes (train/test row counts).",
    )
    args = parser.parse_args()

    labels_df = load_labels_csv(args.labels)
    dataset_df = build_dataset_df(labels_df, tolerance=args.tolerance)
    out_path = save_dataset_csv(dataset_df, args.output)

    print(f"Wrote {len(dataset_df)} rows -> {out_path}")
    print(f"  videos: {len(labels_df)}, tolerance=±{args.tolerance}")

    pos = int(dataset_df["label"].sum()) if len(dataset_df) else 0
    neg = len(dataset_df) - pos
    print(f"  positives={pos}, negatives={neg}")

    if args.print_loo and len(dataset_df):
        vids = unique_video_names(dataset_df)
        print("Leave-one-video-out folds:")
        for train_videos, test_video in iter_leave_one_video_out_splits(vids):
            tr, te = split_by_test_video(dataset_df, test_video)
            print(f"  test={test_video!r}: train_rows={len(tr)}, test_rows={len(te)}")


if __name__ == "__main__":
    main()

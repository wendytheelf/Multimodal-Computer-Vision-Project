"""
Train a classical (hand-crafted feature + Linear SVM) baseline for contact-window classification.

Reads window metadata from ``data/processed/dataset_windows.csv`` (from ``build_dataset.py``),
loads RGB/BGR frames from ``data/raw_videos/<video_name>``, extracts features via
``features_hog.combine_handcrafted_features``, and evaluates with **leave-one-video-out**
cross-validation (splits by **video**, never by individual windows across train/test leakage).

Outputs:
  * ``outputs/metrics/classical_metrics.json``
  * ``outputs/predictions/classical_predictions.csv``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python src/train_classical.py` from the project root (sibling import)
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features_hog import build_hog_descriptor, combine_handcrafted_features

# -----------------------------------------------------------------------------
# Config (defaults relative to project root = parent of ``src/``)
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_windows.csv"
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "spike_clips"

METRICS_JSON = PROJECT_ROOT / "outputs" / "metrics" / "classical_metrics.json"
PREDICTIONS_CSV = PROJECT_ROOT / "outputs" / "predictions" / "classical_predictions.csv"

# Resize frames so width <= this before feature extraction (speed + stable HOG input path)
FEATURE_MAX_WIDTH = 320

# Linear SVM (high-dimensional HOG); adjust C if you over/under-fit
SVM_C = 1.0
SVM_MAX_ITER = 10_000
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Video loading (all frames in memory — fine for short clips)
# -----------------------------------------------------------------------------


def load_all_frames_bgr(video_path: Path) -> list[np.ndarray]:
    """Read every frame from a video as BGR uint8."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    return frames


class VideoFrameCache:
    """
    One-video buffer for triplet access (legacy helper).

    Training no longer uses a dict of **all** decodes: that OOMs with 20+ HD clips.
    See :func:`build_feature_matrix` (single in-flight buffer) for the main path.
    """

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self._name: str | None = None
        self._frames: list[np.ndarray] | None = None

    def get_triplet(self, video_name: str, prev_i: int, curr_i: int, next_i: int) -> list[np.ndarray]:
        if self._name != video_name or self._frames is None:
            self._name = video_name
            path = self.raw_dir / video_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Video not found: {path}. Place raw files in {self.raw_dir}"
                )
            self._frames = load_all_frames_bgr(path)
        all_f = self._frames
        n = len(all_f)
        for idx, name in ((prev_i, "prev_frame"), (curr_i, "curr_frame"), (next_i, "next_frame")):
            if idx < 0 or idx >= n:
                raise IndexError(f"{name}={idx} out of range for {video_name} (n={n})")
        return [all_f[prev_i], all_f[curr_i], all_f[next_i]]


# -----------------------------------------------------------------------------
# Feature matrix
# -----------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    raw_dir: Path,
    hog: cv2.HOGDescriptor,
    *,
    max_width: int,
) -> np.ndarray:
    """
    One row per dataset row: handcrafted feature vector.

    **Memory:** only one full video is kept decoded at a time. When
    ``video_name`` changes between rows, the previous buffer is released.
    This avoids the Linux OOM killer when *many* long HD clips are in the dataset.
    """
    raw_dir = Path(raw_dir)
    rows: list[np.ndarray] = []
    last_name: str | None = None
    all_f: list[np.ndarray] | None = None

    for _, row in df.iterrows():
        name = str(row["video_name"])
        if name != last_name or all_f is None:
            all_f = None
            last_name = name
            path = raw_dir / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Video not found: {path}. Place raw files in {raw_dir}"
                )
            all_f = load_all_frames_bgr(path)

        n = len(all_f)
        prev_i, curr_i, next_i = int(row["prev_frame"]), int(row["curr_frame"]), int(row["next_frame"])
        for idx, label in ((prev_i, "prev_frame"), (curr_i, "curr_frame"), (next_i, "next_frame")):
            if idx < 0 or idx >= n:
                raise IndexError(f"{label}={idx} out of range for {name} (n={n})")
        trip = [all_f[prev_i], all_f[curr_i], all_f[next_i]]
        feat = combine_handcrafted_features(
            trip,
            max_width=max_width,
            hog=hog,
        )
        rows.append(feat)

    return np.stack(rows, axis=0)


# -----------------------------------------------------------------------------
# LOO evaluation
# -----------------------------------------------------------------------------


def leave_one_video_out_eval(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    *,
    svm_c: float,
    svm_max_iter: int,
    random_state: int,
    feature_max_width: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    For each video V: train on all rows with video != V, test on video == V.

    Returns:
        metrics_dict: accuracy, precision, recall, f1 on **all pooled test predictions**.
        predictions_df: one row per sample with scores and labels.
    """
    videos = sorted(df["video_name"].unique())
    if len(videos) < 2:
        raise ValueError(
            "Leave-one-video-out needs at least 2 distinct videos in the dataset. "
            "Add more labeled clips or use a different split strategy for debugging."
        )

    pred_labels: list[int] = []
    scores: list[float] = []
    true_labels: list[int] = []
    meta_video: list[str] = []
    meta_center: list[int] = []

    # Column indices aligned with df rows
    indices = np.arange(len(df))

    for test_video in videos:
        test_mask = (df["video_name"].values == test_video)
        train_mask = ~test_mask

        X_tr = X[train_mask]
        y_tr = y[train_mask]
        X_te = X[test_mask]
        y_te = y[test_mask]
        idx_te = indices[test_mask]

        # Linear SVM + scaling (critical: HOG vs. tiny scalars)
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    LinearSVC(
                        C=svm_c,
                        max_iter=svm_max_iter,
                        random_state=random_state,
                        dual="auto",
                    ),
                ),
            ]
        )
        clf.fit(X_tr, y_tr)

        # Signed distance: positive typically corresponds to class 1 in sklearn's ordering
        dec = clf.decision_function(X_te)
        y_hat = clf.predict(X_te)

        for j in range(len(y_te)):
            pred_labels.append(int(y_hat[j]))
            true_labels.append(int(y_te[j]))
            scores.append(float(dec[j]))
            # map back to row for metadata
            row_idx = int(idx_te[j])
            meta_video.append(str(df.iloc[row_idx]["video_name"]))
            meta_center.append(int(df.iloc[row_idx]["center_frame"]))

    y_true = np.array(true_labels, dtype=int)
    y_pred = np.array(pred_labels, dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_videos_loo": int(len(videos)),
        "feature_max_width": int(feature_max_width),
        "svm_C": float(svm_c),
    }

    pred_df = pd.DataFrame(
        {
            "video_name": meta_video,
            "center_frame": meta_center,
            "true_label": true_labels,
            "pred_label": pred_labels,
            "score": scores,
        }
    )

    return metrics, pred_df


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_predictions(path: Path, pred_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(path, index=False)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classical hand-crafted + Linear SVM baseline (LOO by video).")
    p.add_argument("--dataset", type=Path, default=DATASET_CSV, help="Path to dataset_windows.csv")
    p.add_argument("--raw-dir", type=Path, default=RAW_VIDEO_DIR, help="Folder with raw videos")
    p.add_argument("--metrics-out", type=Path, default=METRICS_JSON, help="Output JSON path")
    p.add_argument("--pred-out", type=Path, default=PREDICTIONS_CSV, help="Output predictions CSV")
    p.add_argument("--max-width", type=int, default=FEATURE_MAX_WIDTH, help="Max width before features")
    p.add_argument("--svm-C", type=float, default=SVM_C, dest="svm_c", help="LinearSVC C")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    feature_max_width = int(args.max_width)
    svm_c = float(args.svm_c)

    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"Dataset CSV not found: {args.dataset}\n"
            "Run: python src/build_dataset.py"
        )

    df = pd.read_csv(args.dataset)
    required = {"video_name", "prev_frame", "curr_frame", "next_frame", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns {missing}: {args.dataset}")

    y = df["label"].astype(int).values
    hog = build_hog_descriptor()

    print(f"Building features for {len(df)} windows from {args.dataset} ...")
    X = build_feature_matrix(df, args.raw_dir, hog, max_width=feature_max_width)
    print(f"Feature matrix shape: {X.shape}")

    print("Leave-one-video-out evaluation ...")
    metrics, pred_df = leave_one_video_out_eval(
        df,
        X,
        y,
        svm_c=svm_c,
        svm_max_iter=SVM_MAX_ITER,
        random_state=RANDOM_STATE,
        feature_max_width=feature_max_width,
    )

    save_metrics(args.metrics_out, metrics)
    save_predictions(args.pred_out, pred_df)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics -> {args.metrics_out}")
    print(f"Wrote predictions -> {args.pred_out}")


if __name__ == "__main__":
    main()

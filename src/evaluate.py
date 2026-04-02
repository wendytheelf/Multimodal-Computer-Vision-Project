"""
Reusable evaluation utilities for volleyball **contact-window** baselines.

Works with prediction CSVs produced by ``train_classical.py`` / ``train_cnn.py``:

  video_name, center_frame, true_label, pred_label, score

**Frame-level metrics** treat each window as an independent binary prediction (sklearn metrics).

**Event-level metrics** treat each **video** as one localization task: pick the ``center_frame``
with the **maximum** ``score`` as the predicted contact frame, then compare to the annotated
``contact_frame`` from ``data/labels/contact_frames.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "contact_frames.csv"
DEFAULT_CLASSICAL_PRED = PROJECT_ROOT / "outputs" / "predictions" / "classical_predictions.csv"

REQUIRED_PRED_COLUMNS = ("video_name", "center_frame", "true_label", "pred_label", "score")


def load_predictions_csv(path: str | Path) -> pd.DataFrame:
    """Load a predictions CSV and validate required columns."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Predictions file not found: {p}")
    df = pd.read_csv(p)
    missing = set(REQUIRED_PRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {p}. Found: {list(df.columns)}")
    return df


def load_contact_labels_csv(path: str | Path) -> pd.DataFrame:
    """
    Load ``contact_frames.csv`` with at least: video_name, contact_frame.

    Extra columns (fps, total_frames) are ignored for localization metrics.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Labels file not found: {p}")
    df = pd.read_csv(p)
    if "video_name" not in df.columns or "contact_frame" not in df.columns:
        raise ValueError(f"Expected video_name and contact_frame in {p}")
    df = df[["video_name", "contact_frame"]].copy()
    df["video_name"] = df["video_name"].astype(str)
    df["contact_frame"] = df["contact_frame"].astype(int)
    return df


def frame_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Standard binary classification metrics per **window** (not per video).

    Positive class is label ``1`` (contact-near window).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_windows": float(len(y_true)),
    }


def frame_level_metrics_from_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    """Convenience wrapper: reads ``true_label`` / ``pred_label`` columns."""
    return frame_level_metrics(
        pred_df["true_label"].values,
        pred_df["pred_label"].values,
    )


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return sklearn confusion matrix (rows=true, cols=pred), shape (2,2) for binary."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def predicted_contact_frame_argmax_score(pred_video_df: pd.DataFrame) -> int:
    """
    For one video's subset of rows, predict contact as the ``center_frame`` with largest ``score``.

    Ties: pandas ``idxmax`` returns the first occurrence of the maximum.
    """
    if pred_video_df.empty:
        raise ValueError("Empty subset: no predictions for this video.")
    idx = pred_video_df["score"].astype(float).idxmax()
    return int(pred_video_df.loc[idx, "center_frame"])


def event_level_metrics(
    pred_df: pd.DataFrame,
    contact_labels: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Per-video localization: argmax score → predicted contact frame vs annotated ``contact_frame``.

    Returns:
        summary: dict with mae, pct_within_2, pct_within_3, n_videos_evaluated
        per_video: DataFrame with columns
            video_name, true_contact_frame, pred_contact_frame, abs_error
    """
    lookup = contact_labels.set_index("video_name")["contact_frame"].to_dict()

    rows: list[dict[str, Any]] = []
    for video_name, sub in pred_df.groupby("video_name", sort=True):
        vkey = str(video_name)
        if vkey not in lookup:
            raise KeyError(
                f"No contact_frame label for video {vkey!r} in labels CSV. "
                "Ensure names match predictions exactly."
            )
        true_contact = int(lookup[vkey])
        pred_contact = predicted_contact_frame_argmax_score(sub)
        abs_err = abs(pred_contact - true_contact)
        rows.append(
            {
                "video_name": vkey,
                "true_contact_frame": true_contact,
                "pred_contact_frame": pred_contact,
                "abs_error": int(abs_err),
            }
        )

    per_video = pd.DataFrame(rows)
    if per_video.empty:
        return {
            "mean_abs_error": float("nan"),
            "pct_within_2_frames": float("nan"),
            "pct_within_3_frames": float("nan"),
            "n_videos_evaluated": 0.0,
        }, per_video

    errors = per_video["abs_error"].values.astype(float)
    summary = {
        "mean_abs_error": float(np.mean(errors)),
        "pct_within_2_frames": float(100.0 * np.mean(errors <= 2.0)),
        "pct_within_3_frames": float(100.0 * np.mean(errors <= 3.0)),
        "n_videos_evaluated": float(len(per_video)),
    }
    return summary, per_video


def evaluate_predictions(
    pred_df: pd.DataFrame,
    contact_labels: pd.DataFrame,
) -> dict[str, Any]:
    """
    Full report: frame-level metrics + confusion matrix counts + event-level summary.

    Returns one JSON-serializable dict (NumPy types cast to float/int).
    """
    frame = frame_level_metrics_from_predictions(pred_df)
    cm = confusion_counts(
        pred_df["true_label"].values,
        pred_df["pred_label"].values,
    )
    event_summary, per_video = event_level_metrics(pred_df, contact_labels)

    return {
        "frame_level": frame,
        "confusion_matrix": {
            "labels_order": [0, 1],
            "counts": cm.tolist(),  # [[TN, FP],[FN, TP]] style with labels [0,1]
        },
        "event_level": event_summary,
        "event_level_per_video": per_video.to_dict(orient="records"),
    }


def save_evaluation_json(report: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    """
    Example CLI: load one predictions file + labels, print metrics JSON to stdout.

    Example::

        python src/evaluate.py --pred outputs/predictions/classical_predictions.csv
        python src/evaluate.py --pred outputs/predictions/cnn_predictions.csv --out outputs/metrics/cnn_eval_detail.json
    """
    parser = argparse.ArgumentParser(description="Evaluate baseline prediction CSVs.")
    parser.add_argument("--pred", type=Path, default=DEFAULT_CLASSICAL_PRED, help="Predictions CSV")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="contact_frames.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save full JSON report (frame + event + per-video).",
    )
    args = parser.parse_args()

    pred_df = load_predictions_csv(args.pred)
    labels = load_contact_labels_csv(args.labels)
    report = evaluate_predictions(pred_df, labels)

    # Pretty-print compact summary for terminal use
    print(json.dumps(report["frame_level"], indent=2))
    print(json.dumps(report["event_level"], indent=2))

    if args.out is not None:
        save_evaluation_json(report, args.out)
    print(f"Evaluated: {args.pred}")


if __name__ == "__main__":
    main()

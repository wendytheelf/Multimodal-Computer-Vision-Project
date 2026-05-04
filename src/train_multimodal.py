"""
Check-In 3 — Multimodal late fusion for volleyball contact-frame detection.

Uses the same windows as ``dataset_windows.csv`` (built by ``build_dataset.py``) and the
same leave-one-video-out (LOO) protocol as the classical / CNN baselines.

For each sample (a 3-frame window centered at ``center_frame``) we build per-variant
feature vectors from cached **pose** and **audio** features:

  * pose-only:            wrist_velocity, wrist_acceleration at [prev, curr, next]
  * audio-only:           audio_rms, audio_onset at [prev, curr, next]
  * pose + audio:         concatenation of both
  * pose + audio + cnn:   adds one feature = CNN score for the same (video_name, center_frame),
                          only if ``outputs/predictions/cnn_predictions.csv`` exists.

Classifier:
  * Small / safe for a tiny dataset — logistic regression with ``StandardScaler`` (default).
  * Class weight = "balanced" to handle contact vs. non-contact imbalance.

Outputs (matching ``evaluate.py`` / ``visualize_results.py`` CSV schema):
  * outputs/predictions/multimodal_pose_predictions.csv
  * outputs/predictions/multimodal_audio_predictions.csv
  * outputs/predictions/multimodal_predictions.csv           # pose + audio (primary)
  * outputs/predictions/multimodal_fusion_cnn_predictions.csv  # optional
  * outputs/metrics/multimodal_metrics.json                  # all variants + C2 comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate import (  # noqa: E402
    evaluate_predictions,
    load_contact_labels_csv,
    load_predictions_csv,
)
from features_audio import extract_audio_features_for_video  # noqa: E402
from features_pose import extract_pose_features_for_video  # noqa: E402

# -----------------------------------------------------------------------------
# Paths & config (relative to project root = parent of ``src/``)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_windows.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "contact_frames.csv"
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "spike_clips"

POSE_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "pose"
AUDIO_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "audio"

OUT_PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUT_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "multimodal_metrics.json"

CLASSICAL_PRED = PROJECT_ROOT / "outputs" / "predictions" / "classical_predictions.csv"
CNN_PRED = PROJECT_ROOT / "outputs" / "predictions" / "cnn_predictions.csv"

# Feature columns (must exist in the cached CSVs)
POSE_FEATURES = ("wrist_velocity", "wrist_acceleration")
AUDIO_FEATURES = ("audio_rms", "audio_onset")

# For each sample, use features at [prev_frame, curr_frame, next_frame] — 3 stacked time steps.
WINDOW_POSITIONS = ("prev_frame", "curr_frame", "next_frame")

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Feature caching helpers
# -----------------------------------------------------------------------------
def ensure_pose_and_audio_cache(
    labels_df: pd.DataFrame,
    *,
    raw_dir: Path,
    pose_cache_dir: Path,
    audio_cache_dir: Path,
    overwrite: bool = False,
    pose_model_complexity: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Make sure cached pose + audio feature CSVs exist for every labeled video.

    Returns:
        (pose_by_video, audio_by_video) dicts keyed by ``video_name``.
    """
    pose_cache: dict[str, pd.DataFrame] = {}
    audio_cache: dict[str, pd.DataFrame] = {}

    for _, row in labels_df.iterrows():
        name = str(row["video_name"])
        vp = raw_dir / name
        if not vp.is_file():
            print(f"[skip] missing video: {vp}")
            continue

        print(f"[features] pose: {name}")
        pose_cache[name] = extract_pose_features_for_video(
            vp,
            cache_dir=pose_cache_dir,
            overwrite=overwrite,
            model_complexity=pose_model_complexity,
        )

        print(f"[features] audio: {name}")
        audio_cache[name] = extract_audio_features_for_video(
            vp,
            fps=float(row["fps"]),
            total_frames=int(row["total_frames"]),
            cache_dir=audio_cache_dir,
            overwrite=overwrite,
        )

    return pose_cache, audio_cache


def frame_value_lookup(df_features: pd.DataFrame, frame_idx: int, columns: tuple[str, ...]) -> np.ndarray:
    """
    Return the ``columns`` row values at ``frame`` == frame_idx as a 1D float32 array.

    Clamps to [0, len-1] so boundary frames do not crash.
    """
    n = len(df_features)
    if n == 0:
        return np.zeros(len(columns), dtype=np.float32)
    idx = int(max(0, min(frame_idx, n - 1)))
    row = df_features.iloc[idx]
    return np.asarray([row[c] for c in columns], dtype=np.float32)


# -----------------------------------------------------------------------------
# Feature-matrix construction
# -----------------------------------------------------------------------------
def build_feature_matrix(
    dataset_df: pd.DataFrame,
    pose_cache: dict[str, pd.DataFrame],
    audio_cache: dict[str, pd.DataFrame],
    *,
    use_pose: bool,
    use_audio: bool,
    cnn_scores: dict[tuple[str, int], float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a feature matrix aligned to ``dataset_df`` row order.

    Returns:
        X: shape (N, D) float32
        feature_names: list of length D describing each column (for debugging / writeups)
    """
    if not use_pose and not use_audio and cnn_scores is None:
        raise ValueError("At least one of use_pose / use_audio / cnn_scores must be provided.")

    # Build name list once so train/test always have the same column meaning.
    names: list[str] = []
    if use_pose:
        for pos in WINDOW_POSITIONS:
            for c in POSE_FEATURES:
                names.append(f"pose::{c}@{pos}")
    if use_audio:
        for pos in WINDOW_POSITIONS:
            for c in AUDIO_FEATURES:
                names.append(f"audio::{c}@{pos}")
    if cnn_scores is not None:
        names.append("cnn::score")

    rows: list[np.ndarray] = []
    for _, r in dataset_df.iterrows():
        vid = str(r["video_name"])
        parts: list[np.ndarray] = []

        if use_pose:
            pose_df = pose_cache.get(vid)
            if pose_df is None:
                parts.append(np.zeros(len(WINDOW_POSITIONS) * len(POSE_FEATURES), dtype=np.float32))
            else:
                for pos in WINDOW_POSITIONS:
                    parts.append(frame_value_lookup(pose_df, int(r[pos]), POSE_FEATURES))

        if use_audio:
            audio_df = audio_cache.get(vid)
            if audio_df is None:
                parts.append(np.zeros(len(WINDOW_POSITIONS) * len(AUDIO_FEATURES), dtype=np.float32))
            else:
                for pos in WINDOW_POSITIONS:
                    parts.append(frame_value_lookup(audio_df, int(r[pos]), AUDIO_FEATURES))

        if cnn_scores is not None:
            key = (vid, int(r["center_frame"]))
            parts.append(np.asarray([float(cnn_scores.get(key, 0.0))], dtype=np.float32))

        rows.append(np.concatenate(parts, axis=0))

    X = np.stack(rows, axis=0).astype(np.float32)
    return X, names


# -----------------------------------------------------------------------------
# LOO training for one variant
# -----------------------------------------------------------------------------
def leave_one_video_out_train(
    dataset_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Train logistic regression with LOO over unique videos.

    Returns a predictions DataFrame with the required columns:
      video_name, center_frame, true_label, pred_label, score
    """
    videos = sorted(dataset_df["video_name"].unique())
    if len(videos) < 2:
        raise ValueError(
            f"Leave-one-video-out needs >= 2 videos (got {len(videos)}). "
            "Ensure contact_frames.csv has multiple videos."
        )

    pred_rows: list[dict] = []
    indices = np.arange(len(dataset_df))

    for test_vid in videos:
        test_mask = (dataset_df["video_name"].values == test_vid)
        train_mask = ~test_mask

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        idx_te = indices[test_mask]

        # If the training fold has only one class (very rare with so few positives),
        # fall back to predicting the majority class on every test sample.
        if len(np.unique(y_tr)) < 2:
            majority = int(np.bincount(y_tr.astype(int)).argmax())
            for j in range(len(y_te)):
                row_idx = int(idx_te[j])
                pred_rows.append(
                    {
                        "video_name": str(dataset_df.iloc[row_idx]["video_name"]),
                        "center_frame": int(dataset_df.iloc[row_idx]["center_frame"]),
                        "true_label": int(y_te[j]),
                        "pred_label": majority,
                        "score": float(majority),
                    }
                )
            continue

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]  # probability of class 1 (contact)
        preds = (proba >= 0.5).astype(int)

        for j in range(len(y_te)):
            row_idx = int(idx_te[j])
            pred_rows.append(
                {
                    "video_name": str(dataset_df.iloc[row_idx]["video_name"]),
                    "center_frame": int(dataset_df.iloc[row_idx]["center_frame"]),
                    "true_label": int(y_te[j]),
                    "pred_label": int(preds[j]),
                    "score": float(proba[j]),
                }
            )

    return pd.DataFrame(pred_rows, columns=["video_name", "center_frame", "true_label", "pred_label", "score"])


# -----------------------------------------------------------------------------
# Metrics assembly
# -----------------------------------------------------------------------------
def summarize_report(report: dict) -> dict:
    """Flatten the report dict to a compact comparable summary."""
    frame = report.get("frame_level", {})
    event = report.get("event_level", {})
    return {
        "accuracy": float(frame.get("accuracy", float("nan"))),
        "precision": float(frame.get("precision", float("nan"))),
        "recall": float(frame.get("recall", float("nan"))),
        "f1": float(frame.get("f1", float("nan"))),
        "mean_abs_error": float(event.get("mean_abs_error", float("nan"))),
        "pct_within_2_frames": float(event.get("pct_within_2_frames", float("nan"))),
        "pct_within_3_frames": float(event.get("pct_within_3_frames", float("nan"))),
    }


def print_comparison_table(summary_by_variant: dict[str, dict]) -> None:
    """Print a fixed-width comparison table for the terminal / report."""
    headers = [
        "model",
        "F1",
        "Prec",
        "Rec",
        "MAE",
        "±2 %",
        "±3 %",
    ]
    colw = [26, 7, 7, 7, 7, 7, 7]

    def fmt_row(vals: list[str]) -> str:
        return "".join(v.ljust(w) for v, w in zip(vals, colw))

    print("\n" + fmt_row(headers))
    print(fmt_row(["-" * (w - 1) for w in colw]))
    for name, s in summary_by_variant.items():
        print(
            fmt_row(
                [
                    name[: colw[0] - 1],
                    f"{s['f1']:.3f}",
                    f"{s['precision']:.3f}",
                    f"{s['recall']:.3f}",
                    f"{s['mean_abs_error']:.2f}",
                    f"{s['pct_within_2_frames']:.1f}",
                    f"{s['pct_within_3_frames']:.1f}",
                ]
            )
        )
    print()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check-In 3: multimodal late fusion (pose + audio).")
    p.add_argument("--dataset", type=Path, default=DATASET_CSV, help="Windowed dataset CSV.")
    p.add_argument("--labels", type=Path, default=LABELS_CSV, help="contact_frames.csv")
    p.add_argument("--raw-dir", type=Path, default=RAW_VIDEO_DIR)
    p.add_argument("--pose-cache-dir", type=Path, default=POSE_CACHE_DIR)
    p.add_argument("--audio-cache-dir", type=Path, default=AUDIO_CACHE_DIR)
    p.add_argument("--out-pred-dir", type=Path, default=OUT_PRED_DIR)
    p.add_argument("--out-metrics", type=Path, default=OUT_METRICS)
    p.add_argument(
        "--cnn-pred",
        type=Path,
        default=CNN_PRED,
        help="Optional: CNN predictions CSV; if present, also trains pose+audio+CNN variant.",
    )
    p.add_argument(
        "--classical-pred",
        type=Path,
        default=CLASSICAL_PRED,
        help="Optional: classical predictions CSV; used for comparison table only.",
    )
    p.add_argument("--overwrite-features", action="store_true", help="Rebuild pose/audio caches.")
    p.add_argument(
        "--pose-model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe Pose model_complexity.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"Windowed dataset not found: {args.dataset}\nRun: python src/build_dataset.py"
        )
    if not args.labels.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {args.labels}")

    dataset_df = pd.read_csv(args.dataset)
    labels_df = pd.read_csv(args.labels)

    required_cols = {"video_name", "center_frame", "prev_frame", "curr_frame", "next_frame", "label"}
    missing = required_cols - set(dataset_df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns {missing}: {args.dataset}")

    y = dataset_df["label"].astype(int).to_numpy()

    # 1. Feature cache (build once; cached CSVs reused on subsequent runs)
    pose_cache, audio_cache = ensure_pose_and_audio_cache(
        labels_df,
        raw_dir=args.raw_dir,
        pose_cache_dir=args.pose_cache_dir,
        audio_cache_dir=args.audio_cache_dir,
        overwrite=args.overwrite_features,
        pose_model_complexity=args.pose_model_complexity,
    )

    # 2. Optional CNN scores map
    cnn_scores_map: dict[tuple[str, int], float] | None = None
    if args.cnn_pred.is_file():
        cnn_df = pd.read_csv(args.cnn_pred)
        if {"video_name", "center_frame", "score"}.issubset(cnn_df.columns):
            cnn_scores_map = {
                (str(r["video_name"]), int(r["center_frame"])): float(r["score"])
                for _, r in cnn_df.iterrows()
            }
            print(f"[fusion] loaded {len(cnn_scores_map)} CNN scores from {args.cnn_pred.name}")
        else:
            print(f"[fusion] {args.cnn_pred} missing expected columns; skipping CNN-fusion variant.")

    # 3. Feature matrices per variant
    X_pose, names_pose = build_feature_matrix(
        dataset_df, pose_cache, audio_cache, use_pose=True, use_audio=False
    )
    X_audio, names_audio = build_feature_matrix(
        dataset_df, pose_cache, audio_cache, use_pose=False, use_audio=True
    )
    X_fusion, names_fusion = build_feature_matrix(
        dataset_df, pose_cache, audio_cache, use_pose=True, use_audio=True
    )

    variants: dict[str, tuple[np.ndarray, list[str]]] = {
        "pose_only": (X_pose, names_pose),
        "audio_only": (X_audio, names_audio),
        "pose_audio_fusion": (X_fusion, names_fusion),
    }

    if cnn_scores_map is not None:
        X_cnnfuse, names_cnnfuse = build_feature_matrix(
            dataset_df,
            pose_cache,
            audio_cache,
            use_pose=True,
            use_audio=True,
            cnn_scores=cnn_scores_map,
        )
        variants["pose_audio_cnn_fusion"] = (X_cnnfuse, names_cnnfuse)

    # 4. Train each variant with LOO & evaluate
    args.out_pred_dir.mkdir(parents=True, exist_ok=True)
    contact_labels = load_contact_labels_csv(args.labels)

    variant_pred_paths: dict[str, Path] = {
        "pose_only": args.out_pred_dir / "multimodal_pose_predictions.csv",
        "audio_only": args.out_pred_dir / "multimodal_audio_predictions.csv",
        "pose_audio_fusion": args.out_pred_dir / "multimodal_predictions.csv",
        "pose_audio_cnn_fusion": args.out_pred_dir / "multimodal_fusion_cnn_predictions.csv",
    }

    summary_by_variant: dict[str, dict] = {}
    full_reports: dict[str, dict] = {}

    for variant_name, (X, feat_names) in variants.items():
        print(f"\n[train] {variant_name}  — features: {X.shape[1]}, samples: {X.shape[0]}")
        preds = leave_one_video_out_train(dataset_df, X, y)
        pred_path = variant_pred_paths[variant_name]
        preds.to_csv(pred_path, index=False)
        print(f"[train] {variant_name} -> {pred_path.name}")

        report = evaluate_predictions(preds, contact_labels)
        full_reports[variant_name] = {
            "feature_names": feat_names,
            **report,
        }
        summary_by_variant[variant_name] = summarize_report(report)

    # 5. Compare against C2 baselines by reading their predictions if present
    for tag, pred_path in [("classical", args.classical_pred), ("cnn", args.cnn_pred)]:
        if not pred_path.is_file():
            continue
        try:
            pred_df = load_predictions_csv(pred_path)
            rep = evaluate_predictions(pred_df, contact_labels)
            full_reports[tag] = rep
            summary_by_variant[tag] = summarize_report(rep)
        except Exception as e:  # pragma: no cover - informational
            print(f"[compare] skipped {pred_path.name}: {e}")

    # 6. Save metrics JSON + print comparison
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary_by_variant,
                "reports": full_reports,
            },
            f,
            indent=2,
        )
    print(f"\nMetrics written -> {args.out_metrics}")

    print_comparison_table(summary_by_variant)


if __name__ == "__main__":
    main()

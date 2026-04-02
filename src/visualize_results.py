"""
Visualization helpers for Check-In 2 baselines.

Writes figures under ``outputs/figures/``:

* Confusion matrices from window-level ``true_label`` vs ``pred_label``
* Per-video **score timelines**: ``center_frame`` vs ``score`` with true/predicted contact lines

Requires matplotlib. Uses evaluation helpers to pick the argmax-score contact frame per video.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate import (  # noqa: E402
    evaluate_predictions,
    load_contact_labels_csv,
    load_predictions_csv,
    predicted_contact_frame_argmax_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRED = PROJECT_ROOT / "outputs" / "predictions" / "classical_predictions.csv"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "contact_frames.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    title: str = "Confusion matrix (windows)",
) -> None:
    """
    Plot a 2×2 confusion matrix (labels 0 then 1).

    Uses counts from sklearn ordering with labels=[0,1].
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="w" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_timeline_for_video(
    pred_video_df: pd.DataFrame,
    true_contact: int,
    pred_contact: int,
    out_path: Path,
    *,
    title: str | None = None,
) -> None:
    """
    Line plot: x = center_frame, y = score.

    Vertical lines: true contact frame (solid) and argmax-score prediction (dashed).
    """
    sub = pred_video_df.sort_values("center_frame")
    x = sub["center_frame"].values
    y = sub["score"].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, y, "-o", markersize=2, linewidth=1, label="contact score")
    ax.axvline(true_contact, color="green", linewidth=2, label=f"true contact ({true_contact})")
    ax.axvline(pred_contact, color="red", linestyle="--", linewidth=2, label=f"pred contact ({pred_contact})")
    ax.set_xlabel("Frame index (window center)")
    ax.set_ylabel("Score (higher = more contact-like)")
    ax.set_title(title or str(pred_video_df["video_name"].iloc[0]))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_figures_for_run(
    pred_df: pd.DataFrame,
    contact_labels: pd.DataFrame,
    out_dir: Path,
    *,
    run_name: str = "run",
) -> None:
    """
    Save confusion matrix + one timeline PNG per video appearing in ``pred_df``.

    Filenames:
      * ``confusion_{run_name}.png``
      * ``timeline_{run_name}_{video_stem}.png``
    """
    out_dir = Path(out_dir)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)

    plot_confusion_matrix(
        pred_df["true_label"].values,
        pred_df["pred_label"].values,
        out_dir / f"confusion_{safe}.png",
        title=f"Confusion — {run_name}",
    )

    lookup = contact_labels.set_index("video_name")["contact_frame"].astype(int).to_dict()
    for video_name, sub in pred_df.groupby("video_name", sort=True):
        vkey = str(video_name)
        true_c = int(lookup[vkey])
        pred_c = predicted_contact_frame_argmax_score(sub)
        vstem = Path(vkey).stem
        plot_score_timeline_for_video(
            sub,
            true_c,
            pred_c,
            out_dir / f"timeline_{safe}_{vstem}.png",
            title=f"{vkey} — scores vs frame",
        )


def plot_qualitative_placeholder(out_path: Path, *, title: str = "Qualitative examples") -> None:
    """
    Optional hook: save a simple placeholder figure reminding you to add montages.

    For real qualitative examples, load frames with OpenCV and use ``plt.imshow`` in a notebook.
    """
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "Qualitative panel: stack frames (t-1, t, t+1)\naround true & predicted contact here.",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion + score timelines for a predictions CSV.")
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--name", type=str, default="classical", help="Tag for output filenames")
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Also write qualitative_notes_{name}.png",
    )
    args = parser.parse_args()

    pred_df = load_predictions_csv(args.pred)
    labels = load_contact_labels_csv(args.labels)
    plot_all_figures_for_run(pred_df, labels, args.out_dir, run_name=args.name)

    if args.placeholder:
        plot_qualitative_placeholder(args.out_dir / f"qualitative_notes_{args.name}.png")

    report = evaluate_predictions(pred_df, labels)
    print("Frame-level:", report["frame_level"])
    print("Event-level:", report["event_level"])
    print(f"Figures saved under {args.out_dir}")


if __name__ == "__main__":
    main()

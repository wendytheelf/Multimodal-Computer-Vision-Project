"""
Check-In 3 — Tier 2 temporal-window ablation for the CNN.

Reuses ``leave_one_video_out_cnn`` from ``train_cnn.py`` and sweeps the
``window_size`` hyperparameter while keeping everything else (LOO folds,
hyperparameters, seed, image size) fixed. This isolates the effect of
**temporal context** on the ResNet-18 baseline.

For each ``W`` in ``--window-sizes``:
  * train LOO
  * save predictions to ``outputs/predictions/cnn_w{W}.csv``
  * compute a full report with ``evaluate.py``

Artifacts:
  * ``outputs/metrics/cnn_window_ablation.json`` — per-W metrics + summary
  * ``outputs/figures/window_ablation.png``       — F1 + MAE vs window size
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate import evaluate_predictions, load_contact_labels_csv  # noqa: E402
from train_cnn import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_WORKERS,
    RANDOM_SEED,
    USE_PRETRAINED,
    WEIGHT_DECAY,
    leave_one_video_out_cnn,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_windows.csv"
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "spike_clips"
LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "contact_frames.csv"

OUT_PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUT_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "cnn_window_ablation.json"
OUT_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "window_ablation.png"


def _summary(report: dict) -> dict:
    """Compact {F1, precision, recall, MAE, ±2, ±3} view for plotting / tables."""
    frame = report.get("frame_level", {})
    event = report.get("event_level", {})
    return {
        "f1": float(frame.get("f1", float("nan"))),
        "precision": float(frame.get("precision", float("nan"))),
        "recall": float(frame.get("recall", float("nan"))),
        "mean_abs_error": float(event.get("mean_abs_error", float("nan"))),
        "pct_within_2_frames": float(event.get("pct_within_2_frames", float("nan"))),
        "pct_within_3_frames": float(event.get("pct_within_3_frames", float("nan"))),
    }


def make_comparison_plot(summary_by_w: dict[int, dict], out_path: Path) -> None:
    """Two-axis plot: F1 (left) and MAE (right) vs window size."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ws = sorted(summary_by_w.keys())
    f1s = [summary_by_w[w]["f1"] for w in ws]
    maes = [summary_by_w[w]["mean_abs_error"] for w in ws]
    pct2 = [summary_by_w[w]["pct_within_2_frames"] for w in ws]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    color_f1 = "tab:blue"
    ax1.set_xlabel("Temporal window size (frames)")
    ax1.set_ylabel("Frame-level F1 (↑)", color=color_f1)
    ax1.plot(ws, f1s, marker="o", color=color_f1, label="F1")
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.set_ylim(bottom=0.0)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color_mae = "tab:red"
    ax2.set_ylabel("Event-level MAE (frames, ↓)", color=color_mae)
    ax2.plot(ws, maes, marker="s", linestyle="--", color=color_mae, label="MAE")
    ax2.tick_params(axis="y", labelcolor=color_mae)

    # Annotate pct-within-2 on each point so the figure is self-contained.
    for w, y, p in zip(ws, f1s, pct2):
        ax1.annotate(f"±2: {p:.0f}%", xy=(w, y), xytext=(4, 6), textcoords="offset points", fontsize=8)

    ax1.set_xticks(ws)
    plt.title("CNN temporal window ablation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier 2: temporal window ablation for the CNN baseline.")
    p.add_argument("--dataset", type=Path, default=DATASET_CSV)
    p.add_argument("--raw-dir", type=Path, default=RAW_VIDEO_DIR)
    p.add_argument("--labels", type=Path, default=LABELS_CSV)
    p.add_argument("--out-pred-dir", type=Path, default=OUT_PRED_DIR)
    p.add_argument("--out-metrics", type=Path, default=OUT_METRICS)
    p.add_argument("--out-figure", type=Path, default=OUT_FIGURE)
    p.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[3, 7, 11],
        help="Odd window sizes to compare (default: 3 7 11).",
    )
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    pre = p.add_mutually_exclusive_group()
    pre.add_argument("--pretrained", dest="pretrained", action="store_true")
    pre.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.set_defaults(pretrained=USE_PRETRAINED)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}\nRun: python src/build_dataset.py"
        )
    if not args.labels.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {args.labels}")

    for w in args.window_sizes:
        if w % 2 == 0 or w < 1:
            raise ValueError(f"--window-sizes entries must be positive odd ints; got {w}")

    df = pd.read_csv(args.dataset)
    contact_labels = load_contact_labels_csv(args.labels)

    args.out_pred_dir.mkdir(parents=True, exist_ok=True)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)

    per_window_report: dict[int, dict] = {}
    summary_by_w: dict[int, dict] = {}

    for w in args.window_sizes:
        print(f"\n========== window_size={w} ==========")
        metrics, pred_df = leave_one_video_out_cnn(
            df,
            raw_dir=args.raw_dir,
            image_size=args.image_size,
            pretrained=bool(args.pretrained),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            random_seed=args.seed,
            num_workers=args.num_workers,
            window_size=w,
        )

        pred_path = args.out_pred_dir / f"cnn_w{w}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"[window={w}] predictions -> {pred_path}")

        report = evaluate_predictions(pred_df, contact_labels)
        per_window_report[w] = {"train_metrics": metrics, **report}
        summary_by_w[w] = _summary(report)

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "window_sizes": list(args.window_sizes),
                "summary": {str(k): v for k, v in summary_by_w.items()},
                "reports": {str(k): v for k, v in per_window_report.items()},
            },
            f,
            indent=2,
        )
    print(f"\nMetrics JSON -> {args.out_metrics}")

    make_comparison_plot(summary_by_w, args.out_figure)
    print(f"Comparison plot -> {args.out_figure}")

    headers = ["W", "F1", "Prec", "Rec", "MAE", "±2 %", "±3 %"]
    colw = [4, 7, 7, 7, 7, 7, 7]
    def row(vals: list[str]) -> str:
        return "".join(v.ljust(w) for v, w in zip(vals, colw))

    print("\n" + row(headers))
    print(row(["-" * (w - 1) for w in colw]))
    for w in sorted(summary_by_w):
        s = summary_by_w[w]
        print(
            row(
                [
                    str(w),
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


if __name__ == "__main__":
    main()

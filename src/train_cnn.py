"""
CNN baseline: ResNet-18 on 3-frame grayscale stacks for contact-window classification.

Each sample uses the same rows as ``build_dataset.py`` / ``dataset_windows.csv``:
frames (t-1, t, t+1) from ``data/raw_videos/<video_name>``. Frames are converted to
grayscale and stacked along the channel dimension so the tensor shape is **(3, H, W)**,
which matches ResNet's expected 3 input channels.

**Leave-one-video-out (LOO)** evaluation trains on all videos except one, tests on the
held-out video, and repeats—no leakage across videos.

With only a handful of short clips, a full CNN can **overfit** quickly; treat metrics as
directional unless you add regularization, more data, or stronger augmentation.

Outputs:
  * ``outputs/metrics/cnn_metrics.json``
  * ``outputs/predictions/cnn_predictions.csv``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

# -----------------------------------------------------------------------------
# Allow `python src/train_cnn.py` from project root
# -----------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# -----------------------------------------------------------------------------
# Config (project root = parent of ``src/``)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_windows.csv"
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "spike_clips"
METRICS_JSON = PROJECT_ROOT / "outputs" / "metrics" / "cnn_metrics.json"
PREDICTIONS_CSV = PROJECT_ROOT / "outputs" / "predictions" / "cnn_predictions.csv"

# Image side length for ResNet (224 is standard for ImageNet-pretrained models)
IMAGE_SIZE = 224

# Use ImageNet pretrained weights when True (motion grayscale channels ≠ natural RGB;
# still useful as a strong init, but ``False`` is a fairer "from scratch" baseline)
USE_PRETRAINED = True

# Training hyperparameters (small dataset defaults — expect overfitting if you crank epochs)
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 3e-4  # lower LR when finetuning pretrained; safe default for small data
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

# Workers: 0 avoids multiprocessing issues on some platforms
NUM_WORKERS = 0


# -----------------------------------------------------------------------------
# Video loading (short clips: keep all frames in RAM)
# -----------------------------------------------------------------------------
def load_all_frames_bgr(video_path: Path) -> list[np.ndarray]:
    """Load every frame as BGR uint8."""
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
    """Cache decoded videos by ``video_name`` to avoid reopening files."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self._cache: dict[str, list[np.ndarray]] = {}

    def get_triplet(self, video_name: str, prev_i: int, curr_i: int, next_i: int) -> list[np.ndarray]:
        if video_name not in self._cache:
            path = self.raw_dir / video_name
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}. Expected under {self.raw_dir}")
            self._cache[video_name] = load_all_frames_bgr(path)
        all_f = self._cache[video_name]
        n = len(all_f)
        for idx, name in ((prev_i, "prev_frame"), (curr_i, "curr_frame"), (next_i, "next_frame")):
            if idx < 0 or idx >= n:
                raise IndexError(f"{name}={idx} out of range for {video_name} (n={n})")
        return [all_f[prev_i], all_f[curr_i], all_f[next_i]]


# -----------------------------------------------------------------------------
# Grayscale 3-frame tensor (3-channel image)
# -----------------------------------------------------------------------------
def bgr_triplet_to_gray_stack_tensor(
    frames_bgr: list[np.ndarray],
    image_size: int,
) -> torch.Tensor:
    """
    Convert three BGR frames to a single tensor (3, H, W) with values in [0, 1].

    Channel order: [t-1, t, t+1] grayscale maps stacked as channels.
    """
    if len(frames_bgr) != 3:
        raise ValueError(f"Expected 3 BGR frames, got {len(frames_bgr)}")
    chans: list[np.ndarray] = []
    for bgr in frames_bgr:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
        chans.append(gray.astype(np.float32) / 255.0)
    x = np.stack(chans, axis=0)  # (3, H, W)
    return torch.from_numpy(x)


# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------
class VolleyballWindowDataset(Dataset):
    """
    One row of ``dataset_windows.csv`` → one (x, y) pair.

    ``x``: float32 tensor (3, image_size, image_size)
    ``y``: int64 scalar label 0/1
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache: VideoFrameCache,
        *,
        image_size: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.cache = cache
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        trip = self.cache.get_triplet(
            str(row["video_name"]),
            int(row["prev_frame"]),
            int(row["curr_frame"]),
            int(row["next_frame"]),
        )
        x = bgr_triplet_to_gray_stack_tensor(trip, self.image_size)
        y = torch.tensor(int(row["label"]), dtype=torch.int64)
        return x, y


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def build_resnet18_binary(*, pretrained: bool) -> nn.Module:
    """
    ResNet-18 with final layer replaced for 2-way (non-contact vs contact) classification.

    Args:
        pretrained: If True, load ImageNet weights (all 3 input channels use the same
            conv1 filters — compatible with our 3-channel input shape).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 2)
    return model


# -----------------------------------------------------------------------------
# Metrics (reusable for any y_true / y_pred / score)
# -----------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy, precision, recall, F1 (positive class = 1)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def class_weights_from_labels(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss (stabilizes tiny imbalanced folds)."""
    labels = labels.astype(int)
    n = len(labels)
    n0 = max(1, int(np.sum(labels == 0)))
    n1 = max(1, int(np.sum(labels == 1)))
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32)


# -----------------------------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y_true, logits) as numpy arrays for the full ``loader`` (no shuffle).

    logits shape: (N, 2)
    """
    model.eval()
    y_list: list[np.ndarray] = []
    logit_list: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy()
        logit_list.append(logits)
        y_list.append(yb.numpy())
    y_true = np.concatenate(y_list, axis=0)
    logits_all = np.concatenate(logit_list, axis=0)
    return y_true, logits_all


def logits_to_pred_and_score(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict class from logits; ``score`` = softmax probability of class 1 (contact).

    Returns:
        pred_label: int array 0/1
        score: float array in (0, 1) unless 2 classes degenerate
    """
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    prob = exp / np.sum(exp, axis=1, keepdims=True)
    pred = np.argmax(logits, axis=1).astype(int)
    score = prob[:, 1].astype(float)
    return pred, score


# -----------------------------------------------------------------------------
# LOO: one fold = hold out one video
# -----------------------------------------------------------------------------
def leave_one_video_out_cnn(
    df: pd.DataFrame,
    *,
    raw_dir: Path,
    image_size: int,
    pretrained: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    random_seed: int,
    num_workers: int,
) -> tuple[dict[str, float | int | bool], pd.DataFrame]:
    """
    Train/eval with LOO by ``video_name``; pool all test predictions for global metrics.
    """
    videos = sorted(df["video_name"].unique())
    if len(videos) < 2:
        raise ValueError(
            "Leave-one-video-out needs at least 2 videos. Add more clips or use a "
            "single train/val split for debugging."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    all_true: list[int] = []
    all_pred: list[int] = []
    all_score: list[float]
    all_score = []
    all_video: list[str] = []
    all_center: list[int] = []

    cache = VideoFrameCache(raw_dir)

    for test_vid in videos:
        train_df = df[df["video_name"] != test_vid].copy()
        test_df = df[df["video_name"] == test_vid].copy()

        train_ds = VolleyballWindowDataset(train_df, cache, image_size=image_size)
        test_ds = VolleyballWindowDataset(test_df, cache, image_size=image_size)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        y_tr = train_df["label"].astype(int).values
        cw = class_weights_from_labels(y_tr).to(device)

        model = build_resnet18_binary(pretrained=pretrained).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=cw)

        for _ in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        y_te, logits = predict_logits(model, test_loader, device)
        pred, score = logits_to_pred_and_score(logits)

        all_true.extend([int(v) for v in y_te])
        all_pred.extend([int(v) for v in pred])
        all_score.extend([float(v) for v in score])

        for _, row in test_df.iterrows():
            all_video.append(str(row["video_name"]))
            all_center.append(int(row["center_frame"]))

    y_true_np = np.array(all_true, dtype=int)
    y_pred_np = np.array(all_pred, dtype=int)
    metrics = classification_metrics(y_true_np, y_pred_np)
    metrics.update(
        {
            "n_samples": int(len(y_true_np)),
            "n_videos_loo": int(len(videos)),
            "image_size": int(image_size),
            "pretrained": bool(pretrained),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "device": str(device),
        }
    )

    pred_df = pd.DataFrame(
        {
            "video_name": all_video,
            "center_frame": all_center,
            "true_label": all_true,
            "pred_label": all_pred,
            "score": all_score,
        }
    )
    return metrics, pred_df


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet-18 CNN baseline with LOO by video.")
    p.add_argument("--dataset", type=Path, default=DATASET_CSV)
    p.add_argument("--raw-dir", type=Path, default=RAW_VIDEO_DIR)
    p.add_argument("--metrics-out", type=Path, default=METRICS_JSON)
    p.add_argument("--pred-out", type=Path, default=PREDICTIONS_CSV)
    pre = p.add_mutually_exclusive_group()
    pre.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use ImageNet weights (default).")
    pre.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train ResNet-18 from random init (no ImageNet weights).",
    )
    p.set_defaults(pretrained=USE_PRETRAINED)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}\nRun: python src/build_dataset.py"
        )

    df = pd.read_csv(args.dataset)
    required = {"video_name", "prev_frame", "curr_frame", "next_frame", "label", "center_frame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns {missing}: {args.dataset}")

    print(
        "Starting CNN LOO (this retrains from scratch each fold). "
        "Small data → high overfitting risk; use metrics as a baseline only.\n"
        f"  device will be: {'cuda' if torch.cuda.is_available() else 'cpu'}\n"
        f"  pretrained={args.pretrained}, image_size={args.image_size}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}"
    )

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
    )

    save_json(args.metrics_out, metrics)
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.pred_out, index=False)

    print(json.dumps({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")}, indent=2))
    print(f"Wrote metrics -> {args.metrics_out}")
    print(f"Wrote predictions -> {args.pred_out}")


if __name__ == "__main__":
    main()

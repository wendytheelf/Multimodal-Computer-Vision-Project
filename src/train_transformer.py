"""
Lightweight **temporal Transformer** baseline for volleyball contact-window classification.

Advanced-extension angle (vs existing baselines):
  * **Classical (HOG+SVM)** — bags handcrafted descriptors over the window but treats them as one flat vector (implicit correlation across time steps).
  * **CNN (ResNet)** — learns spatial appearance per grayscale stack but uses channels as pseudo-time without explicit temporal mixing beyond conv depth.
  * **Multimodal LogReg** — concatenates pose+audio features at prev/curr/next into one flat vector; **no cross-step interaction** beyond what the linear kernel mixes.

This script models the **same** cached pose+audio slice as ``train_multimodal.py`` as an explicit sequence [prev → curr → next]:
  each timestep is a small vector (wrist kinematics + audio descriptors). A tiny **Transformer encoder**
  learns pairwise interactions between timesteps (attention), then we classify from a pooled representation.

Inputs are **tabular frame-aligned features** only (Option B): **no raw pixels**, so RAM stays modest and Colab-friendly.
Optional ``--cnn-pred`` adds one scalar channel (**CNN score at ``center_frame``**, broadcast across the three steps)
for a shallow Option-C-style cue without implementing another backbone here.

Training/evaluation follows **leave-one-video-out** on ``dataset_windows.csv``, consistent with the rest of the repo.

Outputs:
  * ``outputs/predictions/transformer_predictions.csv``
  * ``outputs/metrics/transformer_metrics.json``

CLI uses ``parse_known_args()`` so Jupyter / Colab kernels that append ``-f kernel.json`` do not break execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate import evaluate_predictions, load_contact_labels_csv  # noqa: E402
from train_multimodal import (  # noqa: E402
    AUDIO_FEATURES,
    POSE_FEATURES,
    WINDOW_POSITIONS,
    ensure_pose_and_audio_cache,
    frame_value_lookup,
    summarize_report,
)

# -----------------------------------------------------------------------------
# Paths (match multimodal / CNN defaults)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_windows.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "contact_frames.csv"
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "spike_clips"
POSE_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "pose"
AUDIO_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "audio"

OUT_PRED = PROJECT_ROOT / "outputs" / "predictions" / "transformer_predictions.csv"
OUT_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "transformer_metrics.json"
CNN_PRED_DEFAULT = PROJECT_ROOT / "outputs" / "predictions" / "cnn_predictions.csv"

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Sequence construction — Option B (+ optional CNN broadcast per timestep)
# -----------------------------------------------------------------------------
def load_cnn_score_map(path: Path | None) -> dict[tuple[str, int], float] | None:
    if path is None or not Path(path).is_file():
        return None
    df = pd.read_csv(path)
    if not {"video_name", "center_frame", "score"}.issubset(df.columns):
        print(f"[transformer] skip CNN channel: missing columns in {path}")
        return None
    return {
        (str(r["video_name"]), int(r["center_frame"])): float(r["score"])
        for _, r in df.iterrows()
    }


def build_pose_audio_sequences(
    dataset_df: pd.DataFrame,
    pose_cache: dict[str, pd.DataFrame],
    audio_cache: dict[str, pd.DataFrame],
    *,
    cnn_scores: dict[tuple[str, int], float] | None,
) -> tuple[np.ndarray, int]:
    """
    Stack pose+audio features into ``X_seq`` with shape ``(N, seq_len, feat_dim)``.

    ``seq_len`` = len(WINDOW_POSITIONS) == 3 (prev, curr, next).
    ``feat_dim`` = len(POSE_FEATURES) + len(AUDIO_FEATURES) [+ 1 if CNN broadcast].

    Uses the same lookups as multimodal fusion — **cached CSVs only**, no video decode here.
    """
    seq_len = len(WINDOW_POSITIONS)
    base_dim = len(POSE_FEATURES) + len(AUDIO_FEATURES)
    extra = 1 if cnn_scores is not None else 0
    feat_dim = base_dim + extra

    rows: list[np.ndarray] = []
    for _, r in dataset_df.iterrows():
        vid = str(r["video_name"])
        pose_df = pose_cache.get(vid)
        audio_df = audio_cache.get(vid)

        steps: list[np.ndarray] = []
        for pos in WINDOW_POSITIONS:
            pi = np.zeros(base_dim, dtype=np.float32)
            if pose_df is not None:
                pi[: len(POSE_FEATURES)] = frame_value_lookup(pose_df, int(r[pos]), POSE_FEATURES)
            if audio_df is not None:
                pi[len(POSE_FEATURES) :] = frame_value_lookup(audio_df, int(r[pos]), AUDIO_FEATURES)

            if cnn_scores is not None:
                key = (vid, int(r["center_frame"]))
                cnn_val = float(cnn_scores.get(key, 0.0))
                pi = np.concatenate([pi, np.asarray([cnn_val], dtype=np.float32)], axis=0)
            steps.append(pi)

        seq = np.stack(steps, axis=0)  # (seq_len, feat_dim)
        rows.append(seq)

    X = np.stack(rows, axis=0).astype(np.float32)
    assert X.shape[1] == seq_len and X.shape[2] == feat_dim
    return X, feat_dim


# -----------------------------------------------------------------------------
# Model — compact Transformer encoder + pooling classifier
# -----------------------------------------------------------------------------
class TemporalTransformerClassifier(nn.Module):
    """
    ``feat_dim``-dim inputs per timestep → ``d_model`` → Transformer encoder → mean pool → 2 logits.

    Small enough for Colab GPU (also runs fine on CPU for this dataset size).
    """

    def __init__(
        self,
        *,
        seq_len: int,
        feat_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=num_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feat_dim)
        h = self.input_proj(x)
        h = h + self.pos_embedding[:, : h.size(1), :]
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    feat_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    model_kwargs: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train Transformer on one fold; return test logits (N_te, 2) and softmax positive prob (N_te,).

    Applies ``StandardScaler`` **flattened** across ``seq_len * feat_dim`` fit on train only.
    """
    torch.manual_seed(seed)

    n_train = X_train.shape[0]
    flat_tr = X_train.reshape(n_train, -1)
    scaler = StandardScaler()
    flat_tr_s = scaler.fit_transform(flat_tr).astype(np.float32)
    seq_len = X_train.shape[1]
    X_train_s = flat_tr_s.reshape(-1, seq_len, feat_dim)

    n_te = X_test.shape[0]
    flat_te_s = scaler.transform(X_test.reshape(n_te, -1)).astype(np.float32)
    X_test_s = flat_te_s.reshape(-1, seq_len, feat_dim)

    X_t = torch.from_numpy(X_train_s)
    y_t = torch.from_numpy(y_train.astype(np.int64))
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=min(batch_size, max(1, n_train)), shuffle=True, drop_last=False)

    model = TemporalTransformerClassifier(seq_len=seq_len, feat_dim=feat_dim, **model_kwargs).to(device)

    counts = np.bincount(y_train.astype(int), minlength=2)
    w = counts.sum() / (2 * np.maximum(counts, 1))
    weight = torch.tensor([w[0], w[1]], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    model.eval()
    with torch.no_grad():
        te = torch.from_numpy(X_test_s).to(device)
        logits_te = model(te).cpu().numpy()
        prob_pos = torch.softmax(torch.from_numpy(logits_te), dim=1).numpy()[:, 1]

    return logits_te, prob_pos.astype(np.float64)


def leave_one_video_out_transformer(
    dataset_df: pd.DataFrame,
    X_seq: np.ndarray,
    y: np.ndarray,
    *,
    feat_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    model_kwargs: dict,
    seed: int,
) -> pd.DataFrame:
    videos = sorted(dataset_df["video_name"].unique())
    if len(videos) < 2:
        raise ValueError("Need >= 2 videos for LOO.")

    indices = np.arange(len(dataset_df))
    rows_out: list[dict] = []

    for test_vid in videos:
        test_mask = dataset_df["video_name"].values == test_vid
        train_mask = ~test_mask

        X_tr = X_seq[train_mask]
        y_tr = y[train_mask]
        X_te = X_seq[test_mask]
        y_te = y[test_mask]
        idx_te = indices[test_mask]

        if len(np.unique(y_tr)) < 2:
            majority = int(np.bincount(y_tr.astype(int)).argmax())
            prob_maj = float(majority)
            for j in range(len(y_te)):
                ri = int(idx_te[j])
                rows_out.append(
                    {
                        "video_name": str(dataset_df.iloc[ri]["video_name"]),
                        "center_frame": int(dataset_df.iloc[ri]["center_frame"]),
                        "true_label": int(y_te[j]),
                        "pred_label": majority,
                        "score": prob_maj,
                    }
                )
            continue

        logits_te, prob_pos = train_one_fold(
            X_tr,
            y_tr,
            X_te,
            feat_dim=feat_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            model_kwargs=model_kwargs,
            seed=seed + hash(test_vid) % 10000,
        )

        preds_cls = logits_te.argmax(axis=1)

        for j in range(len(y_te)):
            ri = int(idx_te[j])
            rows_out.append(
                {
                    "video_name": str(dataset_df.iloc[ri]["video_name"]),
                    "center_frame": int(dataset_df.iloc[ri]["center_frame"]),
                    "true_label": int(y_te[j]),
                    "pred_label": int(preds_cls[j]),
                    "score": float(prob_pos[j]),
                }
            )

    return pd.DataFrame(rows_out, columns=["video_name", "center_frame", "true_label", "pred_label", "score"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal Transformer on pose+audio sequences (LOO).")
    parser.add_argument("--dataset", type=Path, default=DATASET_CSV)
    parser.add_argument("--labels", type=Path, default=LABELS_CSV)
    parser.add_argument("--raw-dir", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--pose-cache-dir", type=Path, default=POSE_CACHE_DIR)
    parser.add_argument("--audio-cache-dir", type=Path, default=AUDIO_CACHE_DIR)
    parser.add_argument("--pred-out", type=Path, default=OUT_PRED)
    parser.add_argument("--metrics-out", type=Path, default=OUT_METRICS)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--pose-model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--cnn-pred", type=Path, default=CNN_PRED_DEFAULT)
    parser.add_argument("--no-cnn-channel", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu-only", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        print("[train_transformer] ignoring unrecognized argv:", unknown)

    if args.d_model % args.nhead != 0:
        raise ValueError(f"--d-model ({args.d_model}) must be divisible by --nhead ({args.nhead}).")

    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not args.labels.is_file():
        raise FileNotFoundError(f"Labels not found: {args.labels}")

    dataset_df = pd.read_csv(args.dataset)
    labels_df = pd.read_csv(args.labels)
    required = {"video_name", "center_frame", "prev_frame", "curr_frame", "next_frame", "label"}
    if not required.issubset(dataset_df.columns):
        raise ValueError(f"Dataset missing columns: {required - set(dataset_df.columns)}")

    y = dataset_df["label"].astype(int).to_numpy()

    pose_cache, audio_cache = ensure_pose_and_audio_cache(
        labels_df,
        raw_dir=args.raw_dir,
        pose_cache_dir=args.pose_cache_dir,
        audio_cache_dir=args.audio_cache_dir,
        overwrite=args.overwrite_features,
        pose_model_complexity=args.pose_model_complexity,
    )

    cnn_map = None
    if not args.no_cnn_channel:
        cnn_map = load_cnn_score_map(args.cnn_pred)

    X_seq, feat_dim = build_pose_audio_sequences(
        dataset_df, pose_cache, audio_cache, cnn_scores=cnn_map
    )
    print(f"[transformer] sequence tensor shape: {X_seq.shape}  (feat_dim={feat_dim})")
    print(
        "[transformer] input channels: pose+wrist kinematics + audio RMS/onset per timestep"
        + (" + CNN score broadcast" if cnn_map else "")
        + "; temporal mixing via Transformer encoder."
    )

    device = torch.device("cpu")
    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device("cuda")

    model_kwargs = dict(
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        dim_feedforward=int(args.dim_ff),
        dropout=float(args.dropout),
    )

    print(f"[transformer] device={device}  encoder_layers={args.num_layers}  heads={args.nhead}")
    preds_df = leave_one_video_out_transformer(
        dataset_df,
        X_seq,
        y,
        feat_dim=feat_dim,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        model_kwargs=model_kwargs,
        seed=RANDOM_STATE,
    )

    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(args.pred_out, index=False)
    print(f"[transformer] wrote predictions -> {args.pred_out}")

    contact_labels = load_contact_labels_csv(args.labels)
    report = evaluate_predictions(preds_df, contact_labels)

    # ROC-AUC on pooled LOO predictions (need both classes present)
    try:
        yt = preds_df["true_label"].values.astype(int)
        sc = preds_df["score"].values.astype(np.float64)
        if len(np.unique(yt)) >= 2:
            roc = float(roc_auc_score(yt, sc))
        else:
            roc = float("nan")
    except Exception:
        roc = float("nan")
    report["frame_level"]["roc_auc"] = roc

    summary = summarize_report(report)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "temporal_transformer_pose_audio",
        "sequence_length": len(WINDOW_POSITIONS),
        "feat_dim": feat_dim,
        "cnn_score_channel": cnn_map is not None,
        "transformer_hyperparams": model_kwargs | {"epochs": args.epochs, "batch_size": args.batch_size},
        "device": str(device),
        "summary": summary,
        "full_report": report,
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[transformer] metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()

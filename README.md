# Multimodal Volleyball Spike Contact-Frame Detection

**Author:** Wendy Hu
**Course:** Computer Vision — Final Project
**Status:** Final handoff

A reproducible study of **when the ball makes contact with the hand** during a volleyball spike. The system takes a short attack clip and predicts the **contact frame** by combining video appearance (CNN), pose kinematics (MediaPipe), and audio impact features, evaluated under a strict leave-one-video-out protocol.

---

## 1. Quick navigation

| Doc | Purpose |
|---|---|
| [`final.md`](final.md) | Final write-up: methods, results, interpretation, limitations, next steps |
| [`check-in-1.md`](check-in-1.md) | Milestone 1 — problem framing, dataset plan, system pipeline |
| [`check-in-2.md`](check-in-2.md) | Milestone 2 — classical (HOG + SVM) vs. CNN (ResNet-18) baselines |
| [`check-in-3.md`](check-in-3.md) | Milestone 3 — multimodal late fusion (pose + audio) and temporal Transformer |
| [`data/README.md`](data/README.md) | Data access, layout, and labeling instructions |

---

## 2. Project summary

**Goal.** Estimate the **ball-contact frame** in short volleyball-spike video clips and compare progressively richer methods under one shared protocol.

**Task definition (binary classification over short windows).** For each video, every center frame `t` produces a 3-frame window `(t-1, t, t+1)`. The window is **positive** if `|t − t*| ≤ W` (default `W = 2`), where `t*` is the annotated contact frame, otherwise **negative**. Every model in this repo consumes the same windows, the same labels, and the same leave-one-video-out (LOO) folds, so results are directly comparable.

**Inputs / outputs.**

- **Input:** one short attack clip (`data/spike_clips/V*.mp4`) plus its annotated contact frame in `data/labels/contact_frames.csv`.
- **Frame-level output:** a probability per window (`outputs/predictions/<model>_predictions.csv`).
- **Event-level output:** the predicted contact frame per clip = `argmax score` over windows in that video.

**Methods compared.**

1. **Classical baseline** — hand-crafted features (HOG on the center frame + frame-difference statistics) → Linear SVM. (`src/train_classical.py`)
2. **CNN baseline** — ResNet-18 over a stack of grayscale window frames. (`src/train_cnn.py`)
3. **Multimodal late fusion** — wrist velocity / acceleration (MediaPipe) + audio RMS / onset (librosa) → logistic regression; an optional variant adds the CNN score as one extra feature. (`src/train_multimodal.py`)
4. **Temporal Transformer** — small encoder over the 3 timesteps with self-attention across `[prev → curr → next]`, on the same pose+audio feature vectors (with optional CNN-score channel). (`src/train_transformer.py`)

**Headline outcomes (20-video LOO).**

| Method | Frame F1 | MAE (frames) ↓ | % within ±2 frames ↑ |
|---|---|---|---|
| Classical (HOG + SVM) | 0.186 | — | — |
| CNN (ResNet-18, gray-stack) — 6-video subset | **0.706** | **1.0** | **100%** |
| Pose-only (LogReg) | 0.047 | 68.9 | 0% |
| Audio-only (LogReg) | 0.106 | 20.5 | 40% |
| Pose + Audio (LogReg, **primary fusion**) | 0.108 | 20.6 | 35% |
| Pose + Audio + CNN (LogReg) | 0.136 | 17.0 | **60%** |
| **Temporal Transformer** (pose+audio + CNN channel) | **0.176** | **10.2** | 35% (60% within ±3) |

The CNN gray-stack baseline currently wins on the 6-clip subset where it was last evaluated; the Transformer is the strongest model evaluated on **all 20 clips**. A full discussion (including ROC-AUC, per-video MAE, and qualitative analysis) lives in [`final.md`](final.md). Caveat throughout: **N is small** (20 videos), so report MAE and ±-tolerance percentages alongside F1.

---

## 3. Repository layout

```text
Multimodal-Computer-Vision-Project/
├── README.md                  ← this file
├── final.md                   ← final write-up
├── check-in-1.md              ← milestone 1
├── check-in-2.md              ← milestone 2
├── check-in-3.md              ← milestone 3
├── requirements.txt
├── annotate_contact.py        ← interactive tool to label contact frames
├── data/
│   ├── README.md              ← data access / layout
│   ├── spike_clips/           ← raw .mp4 attack clips (NOT in git, see data/README.md)
│   ├── labels/contact_frames.csv
│   └── processed/
│       ├── dataset_windows.csv          ← built by build_dataset.py
│       └── features/{pose,audio}/*.csv  ← cached per-video features
├── src/
│   ├── build_dataset.py       ← per-video windows + labels CSV
│   ├── features_hog.py        ← HOG + frame-diff features (classical)
│   ├── features_pose.py       ← MediaPipe wrist kinematics (cached)
│   ├── features_audio.py      ← librosa RMS + onset (cached)
│   ├── train_classical.py     ← Linear SVM, LOO
│   ├── train_cnn.py           ← ResNet-18 gray-stack, LOO
│   ├── train_multimodal.py    ← pose / audio / fusion / +CNN, LOO
│   ├── train_transformer.py   ← small temporal Transformer, LOO
│   ├── run_cnn_window_ablation.py
│   ├── evaluate.py            ← shared frame + event metrics
│   ├── visualize_results.py   ← confusion + per-video timelines
│   └── extract_frames.py
├── notebooks/                 ← EDA + script-form milestones (01–05)
├── models/pose_landmarker_full.task   ← MediaPipe pose model file
├── outputs/
│   ├── predictions/<model>_predictions.csv   ← (video, center_frame, true_label, pred_label, score)
│   ├── metrics/<model>_metrics.json
│   ├── metrics/<model>_eval_detail.json      ← frame + event-level breakdown
│   └── figures/{confusion_*,timeline_*}.png
└── presentation/assets/project_pipeline_diagram.png
```

The `outputs/` directory is the canonical place to read final results and figures. The `notebooks/` directory contains a parallel `.py` walkthrough (`01_eda.py … 05_error_analysis.py`) plus the original Jupyter EDA notebooks.

---

## 4. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU PyTorch, install `torch` / `torchvision` from [pytorch.org](https://pytorch.org/) for your CUDA version *first*, then install the rest.

`librosa` (and its `ffmpeg` dependency) is required for audio features. On Linux:

```bash
sudo apt-get install -y ffmpeg
```

The MediaPipe pose model file (`models/pose_landmarker_full.task`) ships in the repo. If you delete it, re-download from the [MediaPipe pose docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

---

## 5. Data

See [`data/README.md`](data/README.md) for the full layout. In short:

- `data/spike_clips/V*.mp4` — 20 short attack clips (≈3–8 s each). **Not stored in git.** Place your own clips in this folder, or contact the author for the dataset bundle.
- `data/labels/contact_frames.csv` — required label file with columns `video_name, contact_frame, fps, total_frames`. The current version (20 rows) ships with the repo.
- `data/processed/` — generated by `src/build_dataset.py`, `src/features_pose.py`, and `src/features_audio.py`; safe to delete and regenerate.

If you want to label new clips yourself:

```bash
python annotate_contact.py            # walks through clips not yet in the CSV
python annotate_contact.py --all      # re-annotate every clip
```

---

## 6. How to run the full pipeline

All commands are run from the repository root.

### 6.1 Build window-level dataset

```bash
python src/build_dataset.py
# → data/processed/dataset_windows.csv
```

### 6.2 Cache pose + audio features (run once)

```bash
python src/features_pose.py
python src/features_audio.py
# → data/processed/features/{pose,audio}/<video>.csv
```

Use `--overwrite` on either script to rebuild the cache from scratch.

### 6.3 Train all baselines

```bash
python src/train_classical.py        # Linear SVM, LOO
python src/train_cnn.py              # ResNet-18 gray-stack, LOO
python src/train_multimodal.py       # pose / audio / pose+audio (+CNN if predictions exist)
python src/train_transformer.py      # small temporal Transformer
```

### 6.4 Evaluate + visualize

```bash
# Detailed frame + event metrics (one call per model)
python src/evaluate.py \
  --pred outputs/predictions/multimodal_predictions.csv \
  --labels data/labels/contact_frames.csv \
  --out   outputs/metrics/multimodal_eval_detail.json

# Confusion matrix + per-video score timelines
python src/visualize_results.py \
  --pred outputs/predictions/multimodal_predictions.csv \
  --labels data/labels/contact_frames.csv \
  --name multimodal \
  --out-dir outputs/figures
```

Substitute `classical`, `cnn`, `transformer`, `multimodal_pose`, `multimodal_audio`, or `multimodal_fusion_cnn` to evaluate / plot the other models.

### 6.5 One-shot (after data is in place)

```bash
python src/build_dataset.py && \
python src/features_pose.py && \
python src/features_audio.py && \
python src/train_classical.py && \
python src/train_cnn.py && \
python src/train_multimodal.py && \
python src/train_transformer.py
```

Expected runtime on a single GPU: a few minutes for the classical / multimodal / Transformer paths; ~10–20 min for the CNN path on 20 clips at 224×224.

---

## 7. Where to read the results

| Path | Contents |
|---|---|
| `outputs/metrics/<model>_metrics.json` | Pooled LOO frame-level metrics (accuracy, P/R/F1) per model |
| `outputs/metrics/<model>_eval_detail.json` | Confusion matrix + **event-level** MAE and ±-frame tolerance, with per-video breakdown |
| `outputs/predictions/<model>_predictions.csv` | One row per window: `video_name, center_frame, true_label, pred_label, score` |
| `outputs/figures/confusion_<model>.png` | Window-level confusion matrix |
| `outputs/figures/timeline_<model>_<video>.png` | Per-video score timeline with annotated and predicted contact frames |
| `outputs/metrics/multimodal_metrics.json` | Side-by-side comparison of all multimodal variants and the Check-In 2 baselines |

---

## 8. Reproducibility checklist

- Fixed `random_state=42` in classical / multimodal pipelines.
- LOO splits are **by video** (`group = video_name`); no frame leaks across folds.
- Identical windows / labels / tolerances across all models (driven by `dataset_windows.csv`).
- Pose and audio features are deterministic given the cached input video and re-used from disk on subsequent runs.
- ResNet-18 weights default to ImageNet (`pretrained=True`); pass `--no-pretrained` to retrain from scratch.
- Re-running any single model script regenerates only its own predictions / metrics / figures.

---

## 9. Limitations (read first)

- **N = 20 clips** ⇒ LOO means are noisy. Always report **MAE** and **±-frame tolerance** alongside F1.
- **Class imbalance** is severe (≈5 positive windows per clip at `tolerance=±2`). Frame-level *accuracy* is misleading; F1 / ROC-AUC / event-level MAE are the metrics that matter.
- **Camera / mic variability is not controlled.** Pose features are in pixels (no metric calibration); audio backgrounds vary across phones.
- **Annotation noise (±1 frame)** is a sizeable fraction of the tolerance window and bleeds into both training labels and evaluation.

The full failure-mode discussion lives in [`final.md` §5](final.md#5-failure-modes--qualitative-analysis).

---

## 10. Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (player detection prototyping).
- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (wrist landmarks).
- [librosa](https://librosa.org/) (audio RMS / onset features).
- PyTorch + torchvision (ResNet-18; Transformer encoder).
- Volleyball spike clips were collected from practice / game footage; see [`data/README.md`](data/README.md) for distribution notes.

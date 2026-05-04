# Final Write-Up — Multimodal Volleyball Spike Contact-Frame Detection

**Author:** Wendy Hu
**Course:** Computer Vision — Final Project

This document is the final analysis. It assumes you have skimmed [`README.md`](README.md) for setup and that you may want to consult [`check-in-1.md`](check-in-1.md), [`check-in-2.md`](check-in-2.md), and [`check-in-3.md`](check-in-3.md) for milestone-by-milestone context.

---

## Table of contents

1. [Project summary](#1-project-summary)
2. [Task definition](#2-task-definition)
3. [System overview](#3-system-overview)
4. [Methods](#4-methods)
   - [4.1 Classical baseline (HOG + frame-diff + SVM)](#41-classical-baseline-hog--frame-diff--linear-svm)
   - [4.2 CNN baseline (ResNet-18 gray-stack)](#42-cnn-baseline-resnet-18-gray-stack)
   - [4.3 Multimodal late fusion (pose + audio)](#43-multimodal-late-fusion-pose--audio)
   - [4.4 Temporal Transformer over `[prev, curr, next]`](#44-temporal-transformer-over-prev-curr-next)
5. [Evaluation protocol](#5-evaluation-protocol)
6. [Results](#6-results)
7. [Failure modes and qualitative analysis](#7-failure-modes-and-qualitative-analysis)
8. [Limitations](#8-limitations)
9. [Next-step ideas](#9-next-step-ideas)
10. [Reproducibility appendix](#10-reproducibility-appendix)

---

## 1. Project summary

The project predicts the **single ball-contact frame** in a short volleyball-spike clip. This is a foundational measurement for downstream analyses (timing vs. jump apex, rally segmentation, etc.) and is hard for two reasons: contact is a sub-30-ms event, and the labels are scarce (≈1 contact frame per clip).

I treat the task as **binary classification over short temporal windows**, then convert window scores to a single predicted contact frame per video by argmax. Four models are trained and compared under one shared protocol:

- **Classical:** HOG + frame-difference statistics fed to a Linear SVM.
- **CNN:** ResNet-18 over a 3-frame grayscale stack centered at `t`.
- **Multimodal late fusion:** wrist velocity / acceleration (MediaPipe) + audio RMS / onset (librosa) → logistic regression. An optional variant adds the CNN score as one extra feature.
- **Temporal Transformer:** small encoder with self-attention across the three timesteps, on the same per-frame pose+audio features (with optional CNN-score channel broadcast across the sequence).

Across the full 20-clip leave-one-video-out (LOO) evaluation, the **Temporal Transformer** is the strongest end-to-end model on the full dataset (frame F1 = 0.176, MAE = 10.2 frames, 60% of clips within ±3 frames). The CNN gray-stack baseline still wins on the 6-clip subset where it was last evaluated (frame F1 = 0.706, MAE = 1.0 frame, 100% within ±2 frames), but that result needs to be reproduced on the full 20-clip set before it can be claimed cleanly. The **multimodal pose+audio+CNN logistic regression** trades a little F1 for the best overall ±-tolerance behavior (60% within ±2, 70% within ±3, MAE = 17.0 frames).

The strongest signal in this study is *not* a single best model but the **per-video MAE histogram**: a small number of catastrophic clips (notably `V10`, `V13`, `V16`) dominate the mean and motivate most of the next-step ideas in §9.

---

## 2. Task definition

> Given a short volleyball spike clip and a candidate frame `t`, predict whether `t` corresponds to the ball-contact frame (or lies within `±W` frames of it), and aggregate scores across all `t` in a video to localize the contact frame.

**Window construction (`src/build_dataset.py`).** Frames are 0-indexed. For every video and every `t` such that `t-1` and `t+1` exist, we emit one row with center frame `t`, label `y_t = 1` if `|t − t*| ≤ W` (default `W = 2`), else `y_t = 0`. The resulting CSV (`data/processed/dataset_windows.csv`) drives every downstream model.

**Class balance.** With `tolerance=±2`, each clip has 5 positives (one per tolerance offset) vs. ~150–490 negatives. Across the 20-clip set this is roughly 100 positives vs. 4 200 negatives — a 1:42 imbalance.

**Frame-level vs. event-level metrics.**

- *Frame-level:* per-window predictions, scored with sklearn (accuracy / precision / recall / F1 / confusion matrix), pooled across LOO test folds.
- *Event-level:* one prediction per video = `argmax score` over windows in that video. We report mean absolute error (MAE) in frames vs. the annotated contact frame, plus % of clips within ±2 and ±3 frames.

These two views answer different questions ("is each window correctly classified?" vs. "did we localize contact in this clip?") and they often disagree on a small dataset.

---

## 3. System overview

```text
data/spike_clips/V*.mp4
        │
        ▼
build_dataset.py ──► data/processed/dataset_windows.csv
        │                    │
        │                    ├─► train_classical.py   ──► classical_predictions.csv
        │                    │
        │                    ├─► train_cnn.py         ──► cnn_predictions.csv
        │                    │
        │                    └─► train_multimodal.py  ──► multimodal_*_predictions.csv
        │                                  ▲
        ▼                                  │
features_pose.py  ─►  pose CSVs  ──────────┤
features_audio.py ─►  audio CSVs ──────────┤
                                            │
                            train_transformer.py ──► transformer_predictions.csv

                evaluate.py / visualize_results.py
                                │
                                ▼
                  outputs/metrics/, outputs/figures/
```

A static rendering of this pipeline lives at `presentation/assets/project_pipeline_diagram.png`.

---

## 4. Methods

All four methods consume the same windows and labels from `dataset_windows.csv`. They differ only in the feature representation and classifier.

### 4.1 Classical baseline (HOG + frame-diff + Linear SVM)

- Implemented in `src/train_classical.py` and `src/features_hog.py`.
- For each window, frames are resized so width ≤ 320 px, converted to grayscale, and:
  - HOG descriptor on the **center** frame.
  - Frame-difference statistics (mean / std of the absolute difference between adjacent frames in the window) as a coarse motion summary.
  - Concatenated and L2-normalized.
- Classifier: `LinearSVC(C=1.0, max_iter=10_000, class_weight='balanced')` inside a `StandardScaler` pipeline.
- Trained per LOO fold; `decision_function` is exposed as the per-window score.

**Why include this?** It is the cheapest interpretable baseline. If linear margin on simple cues already separates contact-near windows, deeper models earn their compute only by improving on it.

### 4.2 CNN baseline (ResNet-18 gray-stack)

- Implemented in `src/train_cnn.py`.
- Each window is decoded on demand to grayscale frames at 224×224 and stacked along the channel dimension (default `W=3`, i.e. (t-1, t, t+1) → tensor shape (3, 224, 224)).
- Backbone: `torchvision.models.resnet18` with ImageNet weights. `conv1` is rebuilt to accept `W` channels; pretrained weights are averaged across the original 3 RGB channels and tiled across the new channels so the model starts from a sensible point.
- Head: global pooling + 2-class fully connected layer.
- Optimizer: Adam, `lr=1e-4`, `weight_decay=1e-4`, 10 epochs, batch size 8.
- Training uses `class_weight`-style handling via balanced cross-entropy weights (positives are rare).
- LOO over videos.

**Why include this?** It tests whether learned spatio-temporal appearance helps over hand-crafted cues under identical supervision.

### 4.3 Multimodal late fusion (pose + audio)

- Implemented in `src/train_multimodal.py`, with feature extractors in `src/features_pose.py` (MediaPipe pose landmarker → right-wrist `(x, y)` per frame → finite-difference velocity / acceleration with NaN-safe interpolation) and `src/features_audio.py` (librosa RMS energy + onset strength resampled onto each video frame's timestamp `t / fps`).
- Per window, features are gathered at `[prev, curr, next]` and concatenated into one flat vector. The exact feature names are recorded in `outputs/metrics/multimodal_metrics.json` under `feature_names`.
- Four variants are trained in one run:
  1. **pose_only** — wrist velocity, acceleration at `[prev, curr, next]` (6 features).
  2. **audio_only** — RMS, onset at `[prev, curr, next]` (6 features).
  3. **pose_audio_fusion** — concatenation (12 features). **Primary fusion model.**
  4. **pose_audio_cnn_fusion** — adds the CNN score for the same `(video, center_frame)` (13 features), only if `outputs/predictions/cnn_predictions.csv` exists.
- Classifier: `LogisticRegression(class_weight='balanced')` inside a `StandardScaler` pipeline. LOO over videos.

**Why include this?** Pose and audio fail in *different* ways from RGB (motion blur vs. missed landmarks vs. crowd noise), so a simple linear combiner is a low-overfitting way to test whether the modalities are complementary on this dataset.

Pose / audio features are cached to disk under `data/processed/features/{pose,audio}/<video>.csv` so repeated runs only pay the extraction cost once.

### 4.4 Temporal Transformer over `[prev, curr, next]`

- Implemented in `src/train_transformer.py`.
- Each window is presented as an **explicit sequence of length 3**: each timestep is the same per-frame pose + audio vector used by §4.3, plus optionally one scalar CNN score broadcast to every timestep.
- Architecture: linear projection to `d_model=64`, learned positional embedding (length 3), `nn.TransformerEncoder` with `nhead=4`, `num_layers=2`, `dim_feedforward=128`, `dropout=0.2`, mean pooling over timesteps, 2-class head.
- Optimizer: AdamW, 15 epochs, batch size 128, balanced binary cross-entropy.
- LOO over videos.

**Why include this?** It tests whether **explicit attention across the three timesteps** improves on the flat-concatenation logistic regression while staying within the same input modality. Self-attention on `L=3` is overkill in principle, but it lets us add longer windows and CNN-score channels later without changing the framework.

---

## 5. Evaluation protocol

- **Splits:** `GroupKFold` with `n_splits = N_videos`, grouping by `video_name` — i.e. **leave-one-video-out**. No frame from a held-out clip ever appears in training.
- **Frame-level metrics:** `accuracy / precision / recall / f1 / confusion_matrix` from sklearn, pooled across all LOO test windows; for the Transformer we additionally compute pooled ROC-AUC.
- **Event-level metrics (`src/evaluate.py`):** for each held-out video, predicted contact frame = `center_frame` with maximum `score`. Compare to `contact_frame` from `data/labels/contact_frames.csv` and report:
  - **MAE** in frames.
  - **% within ±2 frames** and **% within ±3 frames** of the annotated contact.
- **Per-video tables:** every `<model>_eval_detail.json` and `multimodal_metrics.json` includes a `event_level_per_video` array so individual outliers are visible.

This protocol is identical for all four models, modulo `n_videos`: the older classical / CNN runs were last evaluated on a 6-video subset (`V3..V8`) before more clips were collected; the multimodal LR and Transformer runs use all 20.

---

## 6. Results

### 6.1 Headline table (pooled LOO)

| Model | Frame Acc. | Frame F1 | Precision | Recall | ROC-AUC | MAE (frames) ↓ | % ±2 ↑ | % ±3 ↑ | n_videos | n_windows |
|---|---|---|---|---|---|---|---|---|---|---|
| Classical (HOG + SVM) | 0.929 | 0.186 | 0.129 | 0.330 | — | — | — | — | 20 | 4 061 |
| CNN ResNet-18 (gray-stack) — *6-clip subset* | 0.979 | **0.706** | **0.857** | 0.600 | — | **1.0** | **100%** | 100% | 6 | 704 |
| Pose-only (LogReg) | 0.257 | 0.047 | 0.024 | 0.790 | — | 68.85 | 0% | 0% | 20 | 4 290 |
| Audio-only (LogReg) | 0.785 | 0.106 | 0.059 | 0.550 | — | 20.50 | 40% | 60% | 20 | 4 290 |
| Pose + Audio fusion (LogReg, **primary**) | 0.784 | 0.108 | 0.060 | 0.560 | — | 20.55 | 35% | 60% | 20 | 4 290 |
| Pose + Audio + CNN (LogReg) | 0.843 | 0.136 | 0.078 | 0.530 | — | 16.95 | **60%** | **70%** | 20 | 4 290 |
| **Temporal Transformer** (pose+audio + CNN channel) | 0.880 | **0.176** | 0.106 | 0.520 | **0.728** | **10.20** | 35% | 60% | 20 | 4 061 |

Numbers are pulled directly from `outputs/metrics/{classical,cnn,multimodal,transformer}_metrics.json`; see `outputs/metrics/<model>_eval_detail.json` for confusion matrices and per-video tables.

### 6.2 Reading the table

Three observations matter more than the absolute numbers:

1. **Frame-level accuracy is mostly imbalance.** A constant "no contact" predictor gets ≈97% accuracy. The interesting signals are F1, ROC-AUC, and event-level MAE.
2. **Pose-only is broken on this data.** Pose recall is 0.79 because the model fires on every plausible swing peak; precision collapses to 0.024 and the event-level MAE is 68.9 frames (worse than chance for many clips). Wrist velocity in pixels is camera-scale-dependent and saturates well before contact.
3. **Adding the CNN score helps fusion.** Pose+audio alone barely improves on audio alone — the audio onset already captures most of the per-frame impact signal. Adding the CNN score (`pose_audio_cnn_fusion`) lifts MAE from ~20 to 17 frames and pushes within-±2 from 35% to **60%**, which is the biggest gain from any one architectural change in this study.

### 6.3 Per-video event-level errors (Transformer, 20 clips)

| video | true contact | predicted | abs error | comment |
|---|---|---|---|---|
| V1  | 122 | 116 |  6 |  |
| V2  |  83 |  83 |  0 | exact |
| V3  |  75 |  73 |  2 | within ±2 |
| V4  |  37 |  36 |  1 |  |
| V5  |  61 |   7 | **54** | catastrophic — model latched onto an early audio transient |
| V6  |  58 |  60 |  2 |  |
| V7  |  55 |  58 |  3 |  |
| V8  |  53 |  55 |  2 |  |
| V9  | 243 | 246 |  3 |  |
| V10 | 394 | 390 |  4 |  |
| V11 | 128 | 120 |  8 |  |
| V12 |  68 |  66 |  2 |  |
| V13 | 137 | 133 |  4 |  |
| V14 | 137 | 127 | 10 |  |
| V15 |  87 |  80 |  7 |  |
| V16 | 158 |  73 | **85** | catastrophic — predicted ~1 s before true contact |
| V17 | 130 | 128 |  2 |  |
| V18 |  77 |  74 |  3 |  |
| V19 |  85 |  82 |  3 |  |
| V20 | 124 | 121 |  3 |  |

Two outliers (V5, V16) account for ~70 of the 204 cumulative absolute frames of error. Removing them takes the Transformer's MAE from 10.2 to ≈3.6 frames, which is in the ballpark of the inter-annotator ±1-frame ambiguity. This dominates §7 and §9.

### 6.4 Figures

- `outputs/figures/confusion_<model>.png` — pooled window-level confusion matrix per model.
- `outputs/figures/timeline_<model>_<video>.png` — per-video score-vs-frame curve with a vertical line at the annotated contact frame and another at the argmax. The multimodal and Transformer timelines (`timeline_multimodal_*`, `timeline_transformer_*`) are the most informative; they make it easy to spot whether failure was a noisy peak (V5) or a flat-lined missing peak (V16).

---

## 7. Failure modes and qualitative analysis

These are the failure modes that actually appear in the predictions / figures, not the textbook list.

1. **Loud non-contact transients steal the audio peak.**
   *Symptom:* a single clip with MAE ≫ median (e.g., V5 → 54 frames). The audio score has a higher peak earlier in the clip — typically a jump-step thud or a teammate's clap — and the linear fusion follows it.
   *Where it shows up:* `timeline_multimodal_V5.png`, `timeline_audio_V5.png`.
   *Mitigation:* score smoothing + minimum-distance peak finder, or restricting the search to a temporal prior around the swing.

2. **Missing pose detections flatten wrist velocity.**
   *Symptom:* pose-only model fires on long stretches of high recall but near-zero precision because the wrist is interpolated linearly through fast motion. Velocity peak is shifted backward by 5–10 frames.
   *Where it shows up:* per-video pose timelines for V13, V18.
   *Mitigation:* MediaPipe at higher input resolution, fall back to YOLO-cropped player ROI before running pose.

3. **Camera scale change between clips.**
   *Symptom:* pose features are in pixels; a zoomed-in clip has 2–3× higher wrist velocity than a wide-angle clip with the same swing. LOO punishes this directly because train and test clips are filmed differently.
   *Mitigation:* normalize pose features by torso length per frame, or report features in `(x, y)` coordinates relative to a hip anchor.

4. **CNN gray-stack overfits with N=6.**
   *Symptom:* spectacular metrics on the 6-clip subset (F1 = 0.706, MAE = 1 frame) but the model has not yet been retrained on the full 20-clip set. The big train/test gap is unsurprising for a ResNet-18 with ~11M parameters and ~700 windows.
   *Mitigation:* re-train on 20 clips with stronger augmentation; the table will likely move closer to the Transformer once that is done.

5. **Annotation noise dominates near-best predictions.**
   *Symptom:* predictions like V4: pred=36, true=37. Whether this counts as a 1-frame error depends entirely on the human annotator. When the model is already within ±2 frames of contact, further "improvements" are below the label-noise floor.
   *Mitigation:* second-annotator pass on a subset; report Bland-Altman style agreement.

6. **Class imbalance hides recall problems behind accuracy.**
   *Symptom:* the multimodal LR variants all hover at 78–84% accuracy while having frame-level F1 ≤ 0.14. Reporting accuracy without F1 or per-video MAE would have been misleading.
   *Mitigation:* This is procedural — keep using `class_weight='balanced'` and continue reporting F1 + MAE + ±-tolerance side by side.

---

## 8. Limitations

- **N = 20 clips.** Every aggregate number in §6 has very high variance. A single outlier shifts MAE by several frames. Report standard deviations, or per-video tables, whenever space allows.
- **Pose features are not metrically calibrated.** The same swing in two clips with different framing produces different pose feature magnitudes. Without a torso-length normalization step the model has to relearn camera scale per clip.
- **Audio features are not crowd-noise robust.** Practice gym audio is much quieter and cleaner than match audio; the audio model would likely degrade in louder conditions.
- **The "best" model depends on what you measure.** The Transformer wins on F1 and MAE; the pose+audio+CNN logistic regression wins on % within ±2 / ±3. Both are valid for different downstream uses (timing analysis vs. clip-level localization).
- **No held-out test set independent of LOO.** All hyperparameter choices were made by reading the LOO predictions and figures, so the reported numbers are an *upper bound* on true generalization.
- **No statistical tests.** Differences between similar models (e.g., pose+audio vs. audio-only) are within a few absolute counts of positive windows.

---

## 9. Next-step ideas

In rough order of expected return on time:

1. **Re-run the CNN on all 20 clips** with stronger augmentation (random flip, random temporal jitter inside the tolerance window). Until this is done, the most-impressive CNN row in §6.1 should not be cited as the headline result.
2. **Score-side post-processing.** A 5-frame moving average + minimum-distance peak finder applied to the existing window scores is a 30-line change that is likely to drop MAE on noisy clips like V5 and V16 substantially.
3. **Pose normalization.** Replace pixel velocities with wrist position relative to hip center, normalized by torso length; this directly addresses failure mode §7.3.
4. **Larger temporal context for the CNN / Transformer.** Move from `L = 3` to `L = 7` or `L = 11` frames so the CNN sees the full swing arc and the Transformer has more timesteps to attend over. The infrastructure is in place (`run_cnn_window_ablation.py`).
5. **Better ball localization.** A YOLO-based ball tracker gives an explicit "hand-to-ball distance over time" feature that is more physically meaningful than pose alone. The `yolov8*.pt` weights already in the repo are a starting point.
6. **More labeled data.** 30–40 clips would let LOO become trustworthy. An interactive labeler (`annotate_contact.py`) is in place to make this cheap.
7. **Per-fold metric tables.** Currently LOO predictions are pooled into one CSV; saving per-fold metrics would let us report mean ± std and run paired tests across models.
8. **Demo notebook.** A small end-to-end script that takes one new clip, runs all pipelines, and renders a single overlay figure (pose, audio, fused score, predicted/annotated contact) is the natural final-presentation deliverable.

---

## 10. Reproducibility appendix

### 10.1 Environment

- Python 3.10+ inside a virtualenv.
- `pip install -r requirements.txt` (PyTorch comes from PyPI by default; install the GPU build manually first if you need CUDA).
- `ffmpeg` available on `PATH` (required by `librosa` for `.mp4` audio decoding).

### 10.2 One-shot reproduction

From the repository root:

```bash
python src/build_dataset.py
python src/features_pose.py
python src/features_audio.py
python src/train_classical.py
python src/train_cnn.py
python src/train_multimodal.py
python src/train_transformer.py
python src/evaluate.py --pred outputs/predictions/multimodal_predictions.csv \
                      --labels data/labels/contact_frames.csv \
                      --out outputs/metrics/multimodal_eval_detail.json
python src/visualize_results.py --pred outputs/predictions/multimodal_predictions.csv \
                                --labels data/labels/contact_frames.csv \
                                --name multimodal --out-dir outputs/figures
```

### 10.3 Where every number in §6 came from

| Number | File |
|---|---|
| Classical row | `outputs/metrics/classical_metrics.json`, `outputs/metrics/classical_eval_detail.json` |
| CNN row | `outputs/metrics/cnn_metrics.json`, `outputs/metrics/multimodal_metrics.json::reports.cnn` |
| Pose-only / Audio-only / Fusion / +CNN rows | `outputs/metrics/multimodal_metrics.json` |
| Transformer row | `outputs/metrics/transformer_metrics.json` |
| Per-video table in §6.3 | `outputs/metrics/transformer_metrics.json::full_report.event_level_per_video` |

### 10.4 Random seeds and determinism

- Classical and multimodal pipelines fix `random_state = 42` in scikit-learn estimators.
- The CNN and Transformer use PyTorch defaults; non-determinism from CUDA convolution backends has not been fully constrained, so re-runs may shift F1 by ≈0.01–0.02 absolute.
- LOO splits are deterministic given a sorted list of `video_name`s.

### 10.5 What is *not* checked into git

- The raw `.mp4` clips under `data/spike_clips/` (large; see `data/README.md` for distribution).
- `.venv/`, `__pycache__/`, and the YOLOv8 `.pt` weights (large; auto-downloaded by Ultralytics on demand if you re-run the YOLO prototype path).
- Cached features under `data/processed/features/` are regenerated on demand.

---

*End of write-up.*

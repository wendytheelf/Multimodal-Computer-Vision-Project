# Check-In 3: Advanced Extension — Multimodal Late Fusion (Pose + Audio)

**Course:** Computer Vision — Final Project Milestone
**Project:** Volleyball spike **contact-frame** detection
**Status:** Advanced-extension milestone (still an early-course prototype — small dataset)

---

## 1. What advanced method was added

The advanced extension is **multimodal late fusion** of **video pose** and **audio impact** features for contact-frame classification. Concretely:

- **Pose features** (from MediaPipe): per-frame right-wrist **velocity** and **acceleration** in pixels, NaN-safe interpolated over missed detections. Implemented in `src/features_pose.py`.
- **Audio features** (from librosa): per-frame short-time **RMS energy** and **onset strength**, resampled from the audio hop grid onto each video frame's timestamp `frame / fps`. Implemented in `src/features_audio.py`.
- **Fusion classifier** (`src/train_multimodal.py`): a small, regularized **logistic regression** (with `StandardScaler` and `class_weight="balanced"`) trained with **leave-one-video-out** cross-validation on the **same `dataset_windows.csv`** used by Check-In 2. It outputs predictions in the **same CSV schema** (`video_name, center_frame, true_label, pred_label, score`) so `evaluate.py` and `visualize_results.py` work unchanged.

Four variants are trained in one run:

1. **pose_only** — wrist velocity & acceleration at `[prev, curr, next]`
2. **audio_only** — RMS & onset at `[prev, curr, next]`
3. **pose_audio_fusion** — concatenation (the primary advanced method)
4. **pose_audio_cnn_fusion** *(optional)* — adds the ResNet-18 contact score from Check-In 2 as one extra feature (only activated if `outputs/predictions/cnn_predictions.csv` is present)

Pose and audio features are **cached to disk** under `data/processed/features/…` so repeated runs only pay the extraction cost once.

### 1.1 Temporal Transformer (`src/train_transformer.py`) — sequence-level advanced extension

A second advanced baseline is a **small temporal Transformer encoder** over the same three-frame window **as explicit sequence** `[prev → curr → next]`: each timestep is the **same** pose + audio vector per frame as multimodal fusion (`features_pose.py` / `features_audio.py`), optionally plus **one scalar CNN score** broadcast to every timestep when `outputs/predictions/cnn_predictions.csv` exists.

Compared with:

| Approach | What it does over time |
|---|---|
| Classical | Single flat vector over the window |
| CNN ResNet | Motion via stacked grayscale channels + spatial convs |
| Multimodal LogReg | Concatenate `[prev ‖ curr ‖ next]` — **no attention across steps** |
| **Transformer** | **Self-attention** among the three timesteps, then pooled classification |

Training/evaluation uses **leave-one-video-out** and the same CSV outputs schema as other baselines (`evaluate.py`). Outputs: `outputs/predictions/transformer_predictions.csv`, `outputs/metrics/transformer_metrics.json`.

#### Slide-ready numbers (example run — yours may vary slightly by seed/hardware)

**Temporal Transformer** (`train_transformer.py`, defaults: `d_model=64`, `nhead=4`, `num_layers=2`, `epochs=15`, pose+audio **+ CNN score channel**, CPU/GPU auto):

| Metric | Value |
|---|---|
| Frame-level **accuracy** | 0.880 |
| Frame-level **precision** | 0.106 |
| Frame-level **recall** | 0.520 |
| Frame-level **F1** | 0.176 |
| Frame-level **ROC-AUC** (pooled LOO) | **0.728** |
| Event-level **MAE** (frames) | **10.2** |
| Event-level **% within ±2 frames** | **35.0** |
| Event-level **% within ±3 frames** | **60.0** |

*Interpretation cue:* High accuracy mainly reflects **many negative windows**; F1/recall track **sparse positives**. Compare ROC-AUC and MAE against CNN / multimodal rows in §3.1.

---

## 2. Why multimodal fusion fits this project

Single-modality signals each fail in characteristic ways:

| Modality | Strength | Typical failure |
|---|---|---|
| Video (RGB / gray stacks) | Rich spatial detail | Motion blur; small ball; occlusion at contact |
| Pose kinematics | Strong, interpretable peak at swing / impact | Missed landmarks; camera-relative pixel scale |
| Audio impact | Sharp, unambiguous transient at hand-ball contact | Crowd / court noise; overlapping impacts |

Because these failure modes are **largely uncorrelated**, even a simple linear fusion of their features is expected to improve robustness. This also matches the **stated thesis** of the project (`README.md` §9: *Multimodal Fusion*) and reuses signals already explored in the `01_EDA.ipynb` notebook.

We chose **late fusion with logistic regression** (rather than a larger end-to-end network) because the labeled set is very small (**7 videos**) and small linear models are the safest way to avoid overfitting while still being able to combine complementary signals.

---

## 3. Comparison to Check-In 2

All models share the **same windows, labels, and LOO splits**. Frame-level metrics are computed over pooled test windows; event-level metrics select the argmax `score` within each held-out video as the predicted contact frame and compare it to the annotated `contact_frame`.

### 3.1 Results table

> After running the pipeline (see §5), the exact numbers below are produced by `src/train_multimodal.py` and stored in `outputs/metrics/multimodal_metrics.json`. Paste them from the terminal's comparison table or read the JSON's `summary` section.

| Model | F1 (↑) | Precision (↑) | Recall (↑) | MAE frames (↓) | % ±2 (↑) | % ±3 (↑) |
|---|---|---|---|---|---|---|
| Classical (HOG + SVM, C2) | XX | XX | XX | XX | XX | XX |
| CNN (ResNet-18 gray-stack, C2) | XX | XX | XX | XX | XX | XX |
| **Temporal Transformer** (`train_transformer.py`, pose+audio [+CNN]) | **0.176** | **0.106** | **0.52** | **10.2** | **35.0** | **60.0** |
| **pose_only (C3)** | XX | XX | XX | XX | XX | XX |
| **audio_only (C3)** | XX | XX | XX | XX | XX | XX |
| **pose+audio fusion (C3, primary)** | XX | XX | XX | XX | XX | XX |
| *pose+audio+CNN fusion (C3, optional)* | XX | XX | XX | XX | XX | XX |

Frame-level **accuracy ≈ 0.880** and **ROC-AUC ≈ 0.728** for the Transformer are in `outputs/metrics/transformer_metrics.json` (`summary` / `full_report.frame_level`).

### 3.2 Figures to include

Generate with `src/visualize_results.py` on each predictions CSV:

- `outputs/figures/confusion_multimodal.png` (window-level)
- `outputs/figures/timeline_multimodal_<video>.png` — one per held-out video, showing the fused score vs. frame, with vertical lines for the annotated and predicted contact frames. These directly visualize *where* fusion helps or hurts.

---

## 4. Failure analysis

Failures expected on this task and dataset (discuss the ones that show up in your runs):

1. **Label noise (±1 frame)** — Annotated `contact_frame` can be off by a frame, which is a large fraction of the window width. Watch for multiple near-contact windows flipping label when the annotation shifts.
2. **Missed pose detections** — MediaPipe sometimes misses the wrist during fast motion. Interpolation smooths this, but the velocity curve is flattened where detections were missing.
3. **Audio onsets without contact** — Whistles, shoe squeaks, or voices can create RMS / onset peaks that misalign the audio score away from the true contact frame.
4. **Small ball / motion blur** — Primary CNN failure mode from C2; unchanged for the pose+audio model since it does not see raw pixels directly.
5. **Camera scale variation** — Pose velocity is in **pixels / frame**, not meters / second. A zoom change between clips shifts the decision boundary and hurts LOO generalization.
6. **Class imbalance + very small N** — With only **7 videos** and ~10 positive windows per clip (`tolerance=±2`), a single mislabeled clip can swing LOO averages. Report per-video MAE alongside the pooled number.

Pick 2–3 **concrete qualitative examples** from `outputs/figures/timeline_multimodal_*.png`:

- A clip where fusion fixes a CNN error (audio peak pulls the argmax onto the true contact frame).
- A clip where fusion still fails (e.g., the loudest audio peak is a whistle, not the impact).
- A clip where pose-only alone was already correct.

---

## 5. How to run (reproduces all artifacts)

From the repository root:

```bash
# 0. Install deps (once)
python -m pip install -r requirements.txt

# 1. Check-In 2 pipeline (still required for comparison)
python src/build_dataset.py
python src/train_classical.py
python src/train_cnn.py

# 2. Cache pose + audio features (runs once per video, then re-uses cache)
python src/features_pose.py
python src/features_audio.py

# 3. Train the Check-In 3 multimodal variants (LOO)
python src/train_multimodal.py

# 4. Evaluate + plot the primary fused model
python src/evaluate.py \
  --pred outputs/predictions/multimodal_predictions.csv \
  --labels data/labels/contact_frames.csv \
  --out outputs/metrics/multimodal_eval_detail.json

python src/visualize_results.py \
  --pred outputs/predictions/multimodal_predictions.csv \
  --labels data/labels/contact_frames.csv \
  --name multimodal \
  --out-dir outputs/figures

# 5. Temporal Transformer (sequence pose+audio [+ optional CNN channel], LOO)
python src/train_transformer.py
```

Re-run pose / audio extraction from scratch with `--overwrite` on either feature script.

### Outputs produced

| Path | Contents |
|---|---|
| `data/processed/features/pose/*.csv` | Cached wrist kinematics per video |
| `data/processed/features/audio/*.csv` | Cached RMS + onset per video |
| `outputs/predictions/multimodal_pose_predictions.csv` | Pose-only LOO predictions |
| `outputs/predictions/multimodal_audio_predictions.csv` | Audio-only LOO predictions |
| `outputs/predictions/multimodal_predictions.csv` | **Primary**: pose + audio fusion |
| `outputs/predictions/multimodal_fusion_cnn_predictions.csv` | Optional: pose + audio + CNN |
| `outputs/metrics/multimodal_metrics.json` | All variants + C2 comparison |
| `outputs/figures/timeline_multimodal_*.png` | Per-video score timelines |
| `outputs/predictions/transformer_predictions.csv` | Temporal Transformer LOO predictions |
| `outputs/metrics/transformer_metrics.json` | Transformer metrics + ROC-AUC |

---

## 6. Limitations (small-dataset honesty)

- **N = 7 videos** means LOO averages are noisy; a single mislabeled or atypical clip can shift the reported means.
- **Positive class is tiny** (~5 windows per clip at `tolerance=±2`). Precision/recall swings by a few counts. This is why the writeup reports F1 and event-level MAE together; they fail differently under this regime.
- **Handcrafted features, not end-to-end fusion.** The model does not learn joint representations — it just linearly combines interpretable signals. That is deliberate given the dataset size.
- **Camera / recording variability is not controlled for.** Pose pixel scales and audio backgrounds vary across clips; LOO partially measures this.
- **No temporal smoothing of scores yet.** The argmax over raw window scores can be unstable; a small moving-average / peak-finder could reduce ±1-frame jitter.

---

## 7. Next steps for the final deliverable

Priority order based on expected impact on this task:

1. **More labeled data (↑↑).** Ten to twenty additional clips would make LOO means meaningful and let us report standard deviations.
2. **Temporal CNN / clip model (↑).** Upgrade the ResNet-18 gray-stack to a 3D or (2+1)D CNN on longer clips (e.g., 11–16 frames), then fuse its score with pose+audio. This is the Tier 2 item hinted at in the plan.
3. **Score smoothing + peak detection (↑).** Apply a short moving average and a minimum-distance peak finder before argmax — should reduce MAE without changing any classifier.
4. **Better ball localization (→).** YOLOv8-based ball tracking gives an explicit hand-to-ball distance time series that is more physically meaningful than wrist-only kinematics.
5. **Self-supervised / pseudo-labeled pretraining on unlabeled spikes (→).** Useful mainly if more raw video is collected.
6. **Demo UI.** A small script or notebook that, given one video, runs all pipelines and renders one figure with overlaid pose/audio/fused scores is a natural final-presentation deliverable.

### Risks

- **Label ambiguity dominates small-N evaluation.** Plan for an annotation QA pass (e.g., second annotator on 2 clips) before claiming any win for multimodal fusion.
- **Audio quality varies by camera.** Different phones / microphones could bias audio-based contact estimation; keep per-video error reporting so a single clip does not drive the mean.
- **Overclaiming from LOO.** LOO on 7 videos has high variance; always report MAE and ±frame-tolerance percentages alongside F1.

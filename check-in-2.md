# Check-In 2: Contact-Frame Estimation — Classical vs. CNN Baselines

**Course:** Computer Vision — Final Project Milestone  
**Focus:** Binary classification over short frame windows for volleyball **ball-contact** timing  
**Status:** Early-course milestone (not a finished multimodal system)

---

## 1. Project goal

The long-term aim of this project is to **estimate the ball-contact moment** in short recordings of volleyball attacks, using video (and later, optional audio). For **Check-In 2**, the scope is deliberately narrow:

- **Compare two supervised baselines** on the **same task, labels, and evaluation protocol**:
  - a **classical, non-deep** model using hand-crafted features from short temporal windows; and  
  - a **convolutional neural network (CNN)** baseline that sees the same windowed inputs (or a closely matched tensor representation).
- **Evaluate only contact-frame estimation** framed as **binary classification**: “does this window contain (or tightly align with) contact?”

This milestone is meant to establish reproducible baselines and measurement before adding fusion, temporal models, or audio.

---

## 2. Dataset

Videos are collected as **short attack clips** (order of a few seconds) from practice or game footage. Each clip is treated as one “example universe” of frames.

**Assumed layout:**

| Location | Role |
|----------|------|
| `data/raw_videos/` | Original `.mp4` (or similar) files |
| `data/labels/contact_frames.csv` | Per-video **ground-truth contact frame index** (or time → frame mapping agreed in code) |
| `data/processed/` | Optional cached frames, crops, or precomputed features |

**Label format (conceptual):** one primary **contact frame** \( t^\* \) per video (or per labeled segment). Windows are scored as positive if they fall inside a small tolerance around \( t^\* \) (see Task definition).

**Honest constraints:** The usable dataset for this check-in is **small**. That limits how strongly we can claim generalization; **leave-one-video-out cross-validation** is used to avoid optimistic bias from random splits that leak frames from the same video into both train and test.

---

## 3. Task definition

We frame contact timing as **binary classification over short frame windows**.

> **Given** a short volleyball attack video and a **candidate frame** \( t \), **predict** whether \( t \) corresponds to the **ball-contact frame**, or lies **within a small tolerance** \( \pm W \) frames of the annotated contact (inclusive).

**Procedure:**

1. For each video, load frames (or a fixed stride subsample if needed for speed).
2. For each candidate frame \( t \), extract a **temporal window** of \( L \) consecutive frames centered at \( t \) (or ending at \( t \), fixed in code and unchanged across baselines).
3. **Label** \( y_t = 1 \) if \( |t - t^\*| \le W \), else \( y_t = 0 \).

Thus each **window** is one training or test example. Both the classical and CNN baselines consume the **same** window definitions and **the same** \( y_t \) from `contact_frames.csv`.

**Why this framing fits Check-In 2:** It turns “when is contact?” into a **standard supervised learning** problem with clear positives and negatives, lets us reuse **classification metrics**, and keeps the engineering surface area small (no full sequence decoder required yet). It also aligns with how we will later refine localization (e.g., scanning windows across time and taking argmax probability).

---

## 4. Classical baseline

**Model family:** Non-deep classifier on **hand-crafted features** computed from each window (e.g., scikit-learn: logistic regression, linear SVM, or shallow random forest — one primary choice for the report).

**Features (illustrative — all derivable without a deep backbone):**

- **Motion:** frame differencing or optical-flow statistics (mean / variance of magnitude) within the window.
- **Edge / intensity:** simple histograms or gradient energy to capture swing and blur spikes.
- **Color (optional):** coarse statistics if ball/skin/court contrast helps in some clips.

**Training:** Fit the classifier on windows from **training videos only**. Hyperparameters (e.g., regularization strength) may be tuned with an inner loop, but **the outer evaluation remains leave-one-video-out** on the same split policy as the CNN.

**Role in the rubric:** This baseline answers whether **cheap, interpretable** cues already separate contact-near vs. non-contact windows before committing to a CNN.

---

## 5. CNN baseline

**Model family:** A small **CNN** that maps a fixed-size input tensor to a **binary contact / non-contact** label for that window.

**Input:** A stack of **\( L \)** gray-scale or RGB frames resized to \( H \times W \) (e.g., \( 64 \times 64 \) or \( 112 \times 112 \)), i.e. shape \( (L \cdot C) \times H \times W \) or \( L \times H \times W \times C \) depending on implementation.

**Architecture (illustrative):** Several convolutional blocks with pooling, then a fully connected head and sigmoid or two-class softmax. **Regularization:** dropout, weight decay, early stopping, or light data augmentation (time jitter of the window center within allowed bounds — **without** crossing the contact tolerance boundary in a way that corrupts labels).

**Training:** Same labels, same window length, same tolerance \( W \), and **the same leave-one-video-out** folds as the classical baseline. Training uses only videos in the training fold; the held-out video is evaluated exactly once per outer iteration.

**Role in the rubric:** The CNN tests whether **learned spatio-temporal appearance** improves over the classical feature pipeline under identical supervision and evaluation.

---

## 6. Evaluation plan

**Protocol:** **Leave-one-video-out (LOO) cross-validation.**

- If there are \( N \) labeled videos, we train on \( N-1 \) videos and evaluate on the held-out video, **repeating \( N \) times**.
- **Both baselines** are trained and scored on the **identical folds** and **identical window/label construction**.

**Metrics (reported per fold and aggregated):**

| Metric | Definition / use |
|--------|-------------------|
| **Accuracy** | Overall fraction of correct window predictions (informative only with caveats under imbalance). |
| **Precision / Recall / F1** | For the positive class (contact-near window); primary for skewed positives vs. negatives. |
| **ROC-AUC** | Rank quality of predicted contact probability across windows (useful for scanning). |
| **Balanced accuracy** | Average of sensitivity and specificity; mitigates naive accuracy inflation when negatives dominate. |

**Operational contact localization (optional secondary):** On each held-out video, convert window scores to a **single predicted contact frame** (e.g., argmax of predicted probability over \( t \)) and report **mean absolute frame error (MAE)** vs. \( t^\* \). This connects classification back to “timing” while keeping Check-In 2’s core task window-based.

**Class imbalance:** Most windows are negatives; **F1 and ROC-AUC** are emphasized over raw accuracy. Sampling strategies (class weights, balanced batching, or moderate undersampling of negatives) must be **documented** and applied consistently if used.

---

## 7. Results section placeholder

Results will be filled in after running the full LOO protocol. **Both models use the same labels, windows, tolerance \( W \), and splits.**

**Table: Leave-one-video-out summary (placeholder)**

| Method | Mean F1 (↑) | Mean ROC-AUC (↑) | Mean MAE (frames, ↓) | Notes |
|--------|-------------|-------------------|----------------------|--------|
| Classical (hand-crafted + shallow classifier) | XX | XX | XX | Fixed feature code; e.g., logistic regression |
| CNN baseline | XX | XX | XX | Same windows; comparable input resolution |

Per-fold tables and confusion matrices will live under `outputs/metrics/` and figures under `outputs/figures/`.

---

## 8. Failure analysis

Even with a sound protocol, **contact-frame estimation from single-camera attack clips** is hard. Realistic failure modes include:

- **Motion blur** during fast arm swing: edges and ball boundaries smear; both classical gradients and CNN filters lose fine detail.
- **Small apparent ball size:** the ball may occupy only a few pixels; appearance cues are weak unless zoom or cropping is improved later.
- **Camera angle and scale variation:** the same motion looks different across viewpoints; a model trained on a few clips may not invariantly recognize “contact-like” patterns.
- **Occlusion:** bodies, net, or arms can hide the ball or the hand at the critical instant.
- **Label ambiguity:** the “true” contact frame may differ by a frame or two between human annotators; **±1 frame** disagreement bleeds into both training labels and evaluation.
- **Severe class imbalance:** negatives vastly outnumber positives; naive accuracy can look high while **recall for contact** stays poor unless metrics and training explicitly address imbalance.
- **Very small dataset \( N \):** LOO helps honest assessment but **variance across folds** can be large; reported means should be interpreted with per-fold spread (e.g., standard deviation or min/max) when space allows.

**What we learn for the next milestone:** Which failure modes dominate (e.g., blur vs. label noise) guides whether to invest in **better cropping**, **temporal models**, **hard-negative mining**, or **multi-annotator labels** — not in more complex fusion before baselines are understood.

---

## 9. Conclusion

Check-In 2 intentionally limits scope to **contact-frame estimation** via **window-level binary classification**, enabling a direct comparison between a **classical non-deep baseline** and a **CNN baseline** under **shared labels and leave-one-video-out evaluation**. The goal is not to deliver a production system but to **calibrate difficulty**, **document failure modes**, and **lock in a reproducible protocol** before expanding to richer architectures or multimodal cues.

Preliminary expectations (to be validated by the placeholder results above): the CNN may improve when motion patterns are repeated across clips, but **data size and label noise** may cap gains; the classical baseline remains a valuable sanity check and interpretable reference.

---

## 10. Repository structure and how to run

**Expected top-level layout:**

```text
volleyball-contact-timing/
├── README.md
├── check-in-2.md
├── requirements.txt
├── data/
│   ├── raw_videos/
│   ├── labels/
│   │   └── contact_frames.csv
│   └── processed/
├── notebooks/
├── src/
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── metrics/
└── models/
```

**Setup**

```bash
python -m venv .venv
source .venv/bin/activate                        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For **GPU PyTorch**, install `torch` / `torchvision` from [pytorch.org](https://pytorch.org) for your CUDA version first, then install the rest of the requirements (or keep CPU wheels from PyPI if you do not need CUDA).

**Prerequisites before training**

- `data/labels/contact_frames.csv` — columns include `video_name`, `contact_frame`, `fps`, `total_frames`
- `data/raw_videos/<video_name>` — filenames must match `video_name` in the labels file
- At least **two** labeled videos (leave-one-video-out needs \(N \ge 2\))

**Run order (implemented scripts)**

Run everything from the **repository root** (the folder that contains `src/` and `data/`).

1. **Build window-level dataset metadata** (`data/processed/dataset_windows.csv`):

```bash
python src/build_dataset.py
```

2. **Classical baseline** (hand-crafted features + linear SVM, LOO by video):

```bash
python src/train_classical.py
```

Writes `outputs/metrics/classical_metrics.json` and `outputs/predictions/classical_predictions.csv`.

3. **CNN baseline** (ResNet-18, LOO by video):

```bash
python src/train_cnn.py
```

Optional: train from random init instead of ImageNet weights:

```bash
python src/train_cnn.py --no-pretrained
```

Writes `outputs/metrics/cnn_metrics.json` and `outputs/predictions/cnn_predictions.csv`.

4. **Extended evaluation** (frame-level metrics + **event-level** contact localization: argmax score vs. annotated `contact_frame`):

```bash
python src/evaluate.py \
  --pred outputs/predictions/classical_predictions.csv \
  --labels data/labels/contact_frames.csv \
  --out outputs/metrics/classical_eval_detail.json

python src/evaluate.py \
  --pred outputs/predictions/cnn_predictions.csv \
  --labels data/labels/contact_frames.csv \
  --out outputs/metrics/cnn_eval_detail.json
```

5. **Figures** (confusion matrix + per-video score timelines):

```bash
python src/visualize_results.py \
  --pred outputs/predictions/classical_predictions.csv \
  --labels data/labels/contact_frames.csv \
  --name classical \
  --out-dir outputs/figures

python src/visualize_results.py \
  --pred outputs/predictions/cnn_predictions.csv \
  --labels data/labels/contact_frames.csv \
  --name cnn \
  --out-dir outputs/figures
```

Optional placeholder slide for qualitative panels:

```bash
python src/visualize_results.py --pred outputs/predictions/classical_predictions.csv --name classical --out-dir outputs/figures --placeholder
```

6. **Notebook-style analysis scripts** (optional; also runnable as a bundle):

```bash
python notebooks/01_eda.py
python notebooks/02_build_dataset.py
python notebooks/03_classical_baseline.py
python notebooks/04_cnn_baseline.py
python notebooks/05_error_analysis.py
```

**One-shot pipeline** (after data is in place; re-runs training):

```bash
python src/build_dataset.py && \
python src/train_classical.py && \
python src/train_cnn.py && \
python src/evaluate.py --pred outputs/predictions/classical_predictions.csv --labels data/labels/contact_frames.csv --out outputs/metrics/classical_eval_detail.json && \
python src/evaluate.py --pred outputs/predictions/cnn_predictions.csv --labels data/labels/contact_frames.csv --out outputs/metrics/cnn_eval_detail.json && \
python src/visualize_results.py --pred outputs/predictions/classical_predictions.csv --labels data/labels/contact_frames.csv --name classical --out-dir outputs/figures && \
python src/visualize_results.py --pred outputs/predictions/cnn_predictions.csv --labels data/labels/contact_frames.csv --name cnn --out-dir outputs/figures
```

**Where to read results for this report**

| Output | Contents |
|--------|----------|
| `outputs/metrics/classical_metrics.json`, `cnn_metrics.json` | Window-level accuracy, precision, recall, F1 from LOO |
| `outputs/metrics/*_eval_detail.json` | Frame-level metrics + confusion matrix + **event-level** MAE and % within ±2 / ±3 frames |
| `outputs/predictions/*.csv` | Per-window `true_label`, `pred_label`, `score` |
| `outputs/figures/` | `confusion_*.png`, `timeline_*_*.png` |

`librosa` / `ffmpeg` are only needed if you load audio in EDA; core video baselines use OpenCV only.

---

## Suggested next steps (after baselines)

This repository already includes runnable scripts under `src/` (`build_dataset.py`, `train_classical.py`, `train_cnn.py`, `evaluate.py`, `visualize_results.py`, etc.) and notebook-style helpers under `notebooks/`. Natural extensions for the **final** project (beyond Check-In 2) include:

- **YOLO / person crop** before pose or CNN windows, to reduce background variance.
- **Per-fold metric tables** logged to CSV (today LOO is pooled into one predictions file).
- **ROC-AUC** on window scores (mentioned in the evaluation plan; easy to add on top of saved CSVs).
- **Qualitative montages** (true vs. predicted contact) saved beside timelines in `outputs/figures/`.

The bash commands in **§10** reproduce the current end-to-end Check-In 2 pipeline.

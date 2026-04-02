"""
Hand-crafted features for 3-frame (t-1, t, t+1) volleyball contact windows.

Pipeline (high level):
  1. Convert frames to grayscale (optionally after resizing for speed/stability).
  2. Frame-difference *energy*: summary statistics of pixel changes between neighbors.
  3. Edge *density*: fraction of Canny edge pixels on the center frame.
  4. HOG: Histogram of Oriented Gradients on the center frame (fixed geometry for OpenCV).

All functions are intentionally small so you can swap metrics (e.g. L2 vs L1 diff) in one place.

Assumptions
-----------
* Input frames are BGR ``uint8`` images as returned by OpenCV (``cv2.imread`` / ``VideoCapture``).
* The three frames are already aligned (same resolution). If not, resize before calling
  ``combine_handcrafted_features``.
* Classifier training code should **standardize** features (e.g. ``StandardScaler``) since scales differ
  (HOG bins vs. a fraction in [0, 1]).
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------


def bgr_to_gray(bgr: np.ndarray) -> np.ndarray:
    """
    Convert one BGR image to single-channel grayscale (uint8).

    Uses OpenCV's BGR→GRAY weights (human-perceptual luminance), not a simple average.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {bgr.shape}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def resize_bgr_max_width(bgr: np.ndarray, max_width: int) -> np.ndarray:
    """
    Resize BGR image so width is at most ``max_width``, preserving aspect ratio.

    If the image is already narrower, it is returned unchanged.
    """
    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr
    scale = max_width / float(w)
    new_w = max_width
    new_h = int(round(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Frame difference energy
# ---------------------------------------------------------------------------


def frame_difference_energy(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """
    Scalar energy between two grayscale frames of the same shape.

    Definition: mean absolute difference (robust, cheap).

    Args:
        gray_a, gray_b: 2D uint8 or float arrays, same shape.

    Returns:
        Mean |gray_a - gray_b| over pixels (float).
    """
    if gray_a.shape != gray_b.shape:
        raise ValueError(f"Shape mismatch: {gray_a.shape} vs {gray_b.shape}")
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def pairwise_difference_energies(gray_prev: np.ndarray, gray_curr: np.ndarray, gray_next: np.ndarray) -> np.ndarray:
    """
    Two scalars: energy between (t-1, t) and (t, t+1).

    Returns:
        np.array of shape (2,): [E(t-1,t), E(t,t+1)].
    """
    e1 = frame_difference_energy(gray_prev, gray_curr)
    e2 = frame_difference_energy(gray_curr, gray_next)
    return np.array([e1, e2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Edge density (Canny)
# ---------------------------------------------------------------------------


def edge_density_canny(
    gray: np.ndarray,
    low_thresh: float = 80.0,
    high_thresh: float = 160.0,
) -> float:
    """
    Edge density = fraction of pixels marked as edges by Canny.

    Canny expects uint8 input. Thresholds are fixed defaults; tune per domain if needed.

    Returns:
        Scalar in [0, 1] (approximately; depends on image content).
    """
    if gray.dtype != np.uint8:
        g = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        g = gray
    edges = cv2.Canny(g, int(low_thresh), int(high_thresh))
    return float(np.mean(edges > 0))


# ---------------------------------------------------------------------------
# HOG (OpenCV)
# ---------------------------------------------------------------------------

# Default OpenCV HOG window: 64×128 (width × height). Image must match exactly for ``compute``.
_DEFAULT_HOG_WIN = (64, 128)
_DEFAULT_HOG_BLOCK = (16, 16)
_DEFAULT_HOG_BLOCK_STRIDE = (8, 8)
_DEFAULT_HOG_CELL = (8, 8)
_DEFAULT_HOG_NBINS = 9


def build_hog_descriptor() -> cv2.HOGDescriptor:
    """Create a HOG descriptor with fixed geometry (reusable across calls)."""
    win_size = _DEFAULT_HOG_WIN
    block_size = _DEFAULT_HOG_BLOCK
    block_stride = _DEFAULT_HOG_BLOCK_STRIDE
    cell_size = _DEFAULT_HOG_CELL
    nbins = _DEFAULT_HOG_NBINS
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def hog_features(gray: np.ndarray, hog: cv2.HOGDescriptor | None = None) -> np.ndarray:
    """
    Compute HOG feature vector on a **single** grayscale frame.

    The image is resized to exactly the HOG window size (64×128, width×height).

    Args:
        gray: 2D grayscale (uint8 recommended).
        hog: Optional pre-built ``cv2.HOGDescriptor`` (avoids reallocating).

    Returns:
        1D float32 vector (length depends only on HOG geometry, not input resolution).
    """
    if hog is None:
        hog = build_hog_descriptor()
    w, h = _DEFAULT_HOG_WIN
    # cv2.resize: (width, height)
    small = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    if small.dtype != np.uint8:
        small = np.clip(small, 0, 255).astype(np.uint8)
    vec = hog.compute(small)
    return vec.astype(np.float32).ravel()


# ---------------------------------------------------------------------------
# Full vector
# ---------------------------------------------------------------------------


def combine_handcrafted_features(
    frames_bgr: Sequence[np.ndarray],
    *,
    max_width: int = 320,
    hog: cv2.HOGDescriptor | None = None,
) -> np.ndarray:
    """
    Build one feature vector from three consecutive BGR frames: [t-1, t, t+1].

    Order of concatenation:
      * 2 difference-energy scalars (mean abs diff)
      * 1 edge-density scalar (Canny on center frame)
      * HOG vector from center frame

    Args:
        frames_bgr: Length-3 sequence: previous, current, next (BGR uint8).
        max_width: Resize each frame so width <= this (aspect preserved) before features.
        hog: Optional shared HOG object.

    Returns:
        1D float32 vector.
    """
    if len(frames_bgr) != 3:
        raise ValueError(f"Expected 3 BGR frames, got {len(frames_bgr)}")

    resized = [resize_bgr_max_width(f, max_width) for f in frames_bgr]
    g0, g1, g2 = [bgr_to_gray(f) for f in resized]

    diff_part = pairwise_difference_energies(g0, g1, g2)
    edge_part = np.array([edge_density_canny(g1)], dtype=np.float32)
    hog_part = hog_features(g1, hog=hog)

    return np.concatenate([diff_part, edge_part, hog_part], axis=0)


def feature_dim(hog: cv2.HOGDescriptor | None = None) -> int:
    """Helper: HOG length + 3 scalar handcrafted stats (2 diffs + 1 edge)."""
    h = hog or build_hog_descriptor()
    probe = np.zeros((_DEFAULT_HOG_WIN[1], _DEFAULT_HOG_WIN[0]), dtype=np.uint8)
    return 3 + hog_features(probe, hog=h).shape[0]

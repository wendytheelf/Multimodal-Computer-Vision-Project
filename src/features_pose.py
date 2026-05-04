"""
Pose features from video using MediaPipe **Tasks API** (PoseLandmarker).

For each frame of a video we extract the **spiker's right-wrist** landmark and compute:

  * ``wrist_x`` / ``wrist_y``  — pixel coordinates
  * ``wrist_velocity``         — Euclidean distance to previous frame's wrist (px / frame)
  * ``wrist_acceleration``     — difference of consecutive velocities

When several people appear in the clip (e.g. blockers, defenders, the setter), we
detect up to ``num_poses`` poses per frame, link them across frames with a simple
nearest-neighbour mid-hip tracker, and pick the **track with the highest peak
right-wrist velocity** as the spiker. Frames where the spiker is occluded yield
NaNs that are linearly interpolated downstream so feature windows stay valid.

Features are cached per-video to ``data/processed/features/pose/<video_stem>.csv``.

Why the Tasks API: MediaPipe ≥ 0.10 on Python 3.13 ships **only** the new
``mediapipe.tasks`` API; the legacy ``mediapipe.solutions.pose`` module no longer
exists. The Tasks API needs a ``.task`` model file, which we auto-download to
``models/pose_landmarker_<variant>.task`` on first run.

The 33-landmark indexing matches the legacy API, so ``RIGHT_WRIST_INDEX = 16``.
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "spike_clips"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "pose"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

POSE_FEATURE_COLUMNS = ("wrist_x", "wrist_y", "wrist_velocity", "wrist_acceleration")

# 33-landmark MediaPipe pose indexing (same as the legacy API).
RIGHT_WRIST_INDEX = 16
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

# Default number of poses MediaPipe is allowed to return per frame.
# Using >1 lets us tell the spiker apart from blockers / setters / defenders.
DEFAULT_NUM_POSES = 5

# Default visibility threshold for landmarks we trust.
DEFAULT_VISIBILITY_THRESH = 0.3

# Map "model_complexity" (legacy concept) → Tasks API model file.
# Source: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
TASK_MODEL_URLS: dict[str, str] = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}

# Standard 33-landmark POSE_CONNECTIONS (also identical to the legacy API).
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


# ---------------------------------------------------------------------------
# Model file management
# ---------------------------------------------------------------------------
def _complexity_to_variant(model_complexity: int) -> str:
    """Map legacy 0/1/2 model_complexity values to Tasks API variant names."""
    return {0: "lite", 1: "full", 2: "heavy"}.get(int(model_complexity), "full")


def ensure_pose_task_model(
    *,
    variant: str = "full",
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> Path:
    """
    Ensure ``pose_landmarker_<variant>.task`` exists locally. Downloads on first call.
    Returns the local path to the ``.task`` file.
    """
    if variant not in TASK_MODEL_URLS:
        raise ValueError(f"Unknown pose variant: {variant!r}. Choose from {list(TASK_MODEL_URLS)}")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / f"pose_landmarker_{variant}.task"
    if out.is_file() and out.stat().st_size > 0:
        return out
    url = TASK_MODEL_URLS[variant]
    print(f"[pose] downloading {url} -> {out}")
    urllib.request.urlretrieve(url, str(out))
    return out


# ---------------------------------------------------------------------------
# Lightweight detector wrapper
# ---------------------------------------------------------------------------
class MediaPipePoseDetector:
    """
    Minimal wrapper around the MediaPipe Tasks ``PoseLandmarker`` for video.

    Use as a context manager so the underlying detector is always closed:

        with MediaPipePoseDetector(model_path) as det:
            det.detect_for_video(rgb_frame, timestamp_ms)  # → list of (x, y, z, visibility)
    """

    def __init__(self, model_path: str | Path, *, num_poses: int = 1) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
        self._mp = mp
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def __enter__(self) -> "MediaPipePoseDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass

    def detect_for_video(
        self, frame_rgb: np.ndarray, timestamp_ms: int
    ) -> list[list[tuple[float, float, float, float]]]:
        """
        Run the detector on one RGB frame.

        Returns a list of poses; each pose is a list of (x, y, z, visibility) tuples,
        one per landmark. ``x`` and ``y`` are normalized to [0, 1]. Empty list if
        nothing is detected.
        """
        mp = self._mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect_for_video(mp_image, int(timestamp_ms))
        out: list[list[tuple[float, float, float, float]]] = []
        for pose in (result.pose_landmarks or []):
            out.append(
                [
                    (
                        float(getattr(lm, "x", 0.0)),
                        float(getattr(lm, "y", 0.0)),
                        float(getattr(lm, "z", 0.0)),
                        float(getattr(lm, "visibility", 0.0)),
                    )
                    for lm in pose
                ]
            )
        return out


# ---------------------------------------------------------------------------
# Multi-pose tracking and spiker selection
# ---------------------------------------------------------------------------
@dataclass
class PoseTrack:
    """One person's trajectory across frames (sparse: only frames they appear in)."""

    track_id: int
    frames: list[int] = field(default_factory=list)
    landmarks: list[list[tuple[float, float, float, float]]] = field(default_factory=list)
    last_frame: int = -1
    last_root: tuple[float, float] = (0.0, 0.0)


class MultiPoseTracker:
    """
    Greedy nearest-neighbour tracker on **mid-hip** position (normalized x, y).

    Each call to :meth:`step` consumes the poses MediaPipe returned for one frame and
    assigns each pose to either an existing track (if its mid-hip is within
    ``max_distance`` of a track's last seen mid-hip and the track was seen within
    ``max_gap`` frames) or a new track.

    A pose is ignored (id = -1) if neither hip landmark is visible enough to give us a
    stable root point.
    """

    def __init__(self, *, max_distance: float = 0.15, max_gap: int = 10) -> None:
        self.tracks: dict[int, PoseTrack] = {}
        self._next_id = 0
        self.max_distance = float(max_distance)
        self.max_gap = int(max_gap)

    @staticmethod
    def _root_xy(
        pose: list[tuple[float, float, float, float]],
        *,
        visibility_thresh: float = DEFAULT_VISIBILITY_THRESH,
    ) -> tuple[float, float] | None:
        """Mid-hip in normalized image coords, or None if both hips are unreliable."""
        if len(pose) <= max(LEFT_HIP_INDEX, RIGHT_HIP_INDEX):
            return None
        lh = pose[LEFT_HIP_INDEX]
        rh = pose[RIGHT_HIP_INDEX]
        if lh[3] < visibility_thresh and rh[3] < visibility_thresh:
            return None
        return ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)

    def step(
        self,
        frame_idx: int,
        poses: list[list[tuple[float, float, float, float]]],
    ) -> list[int]:
        """Assign a track id to each pose in this frame (in detection order)."""
        ids: list[int] = []
        claimed: set[int] = set()

        for pose in poses:
            root = self._root_xy(pose)
            if root is None:
                ids.append(-1)
                continue

            best_tid = -1
            best_d = self.max_distance
            for tid, tr in self.tracks.items():
                if tid in claimed:
                    continue
                if frame_idx - tr.last_frame > self.max_gap:
                    continue
                dx = root[0] - tr.last_root[0]
                dy = root[1] - tr.last_root[1]
                d = float(np.hypot(dx, dy))
                if d < best_d:
                    best_d = d
                    best_tid = tid

            if best_tid == -1:
                best_tid = self._next_id
                self._next_id += 1
                self.tracks[best_tid] = PoseTrack(track_id=best_tid)

            tr = self.tracks[best_tid]
            tr.frames.append(int(frame_idx))
            tr.landmarks.append(pose)
            tr.last_frame = int(frame_idx)
            tr.last_root = root
            claimed.add(best_tid)
            ids.append(best_tid)

        return ids


def _track_wrist_velocity_peak(
    track: PoseTrack,
    *,
    total_frames: int,
    video_w: int,
    video_h: int,
    visibility_thresh: float = DEFAULT_VISIBILITY_THRESH,
) -> float:
    """Peak right-wrist speed (px / frame) over the whole clip for one track."""
    xy = np.full((total_frames, 2), np.nan, dtype=np.float32)
    for f, pose in zip(track.frames, track.landmarks):
        if 0 <= f < total_frames and len(pose) > RIGHT_WRIST_INDEX:
            lm = pose[RIGHT_WRIST_INDEX]
            if lm[3] >= visibility_thresh:
                xy[f, 0] = lm[0] * video_w
                xy[f, 1] = lm[1] * video_h

    if not np.any(np.isfinite(xy)):
        return 0.0

    x = interpolate_nan_1d(xy[:, 0])
    y = interpolate_nan_1d(xy[:, 1])
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    v = np.sqrt(dx * dx + dy * dy)
    return float(v.max()) if v.size else 0.0


def select_spiker_track(
    tracker: MultiPoseTracker,
    *,
    total_frames: int,
    video_w: int,
    video_h: int,
    min_track_len: int = 5,
) -> tuple[int | None, float]:
    """
    Pick the track with the highest **peak right-wrist velocity** — that's the spiker.

    Returns ``(track_id, peak_velocity_px_per_frame)``; ``track_id`` is ``None`` if no
    track meets the minimum length requirement.
    """
    best_tid: int | None = None
    best_peak = -1.0
    for tid, tr in tracker.tracks.items():
        if len(tr.frames) < min_track_len:
            continue
        peak = _track_wrist_velocity_peak(
            tr, total_frames=total_frames, video_w=video_w, video_h=video_h
        )
        if peak > best_peak:
            best_peak = peak
            best_tid = tid
    return best_tid, max(best_peak, 0.0)


@dataclass
class TrackedPoseRun:
    """All per-frame pose detections + the tracker's spiker decision for one video."""

    fps: float
    width: int
    height: int
    total_frames: int
    poses_per_frame: list[list[list[tuple[float, float, float, float]]]]
    track_ids_per_frame: list[list[int]]
    spiker_track_id: int | None
    spiker_peak_velocity_px: float


def run_pose_tracking_on_video(
    video_path: str | Path,
    *,
    model_complexity: int = 1,
    model_dir: Path = DEFAULT_MODEL_DIR,
    num_poses: int = DEFAULT_NUM_POSES,
    max_distance: float = 0.15,
    max_gap: int = 10,
    min_track_len: int = 5,
) -> TrackedPoseRun:
    """
    Single MediaPipe pass over the video that captures *all* detected poses, links
    them into tracks, and decides which track is the spiker.

    Returned :class:`TrackedPoseRun` is everything the demo needs to draw (every
    person on the court each frame, with the spiker highlighted) and everything the
    feature-cache needs to extract a clean wrist trajectory.
    """
    import cv2

    variant = _complexity_to_variant(model_complexity)
    task_model = ensure_pose_task_model(variant=variant, model_dir=Path(model_dir))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = MultiPoseTracker(max_distance=max_distance, max_gap=max_gap)
    poses_per_frame: list[list[list[tuple[float, float, float, float]]]] = []
    track_ids_per_frame: list[list[int]] = []

    try:
        with MediaPipePoseDetector(task_model, num_poses=num_poses) as det:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = int(frame_idx * 1000.0 / max(fps, 1e-6))
                poses = det.detect_for_video(rgb, ts_ms)
                ids = tracker.step(frame_idx, poses)
                poses_per_frame.append(poses)
                track_ids_per_frame.append(ids)
                frame_idx += 1
    finally:
        cap.release()

    total_frames = len(poses_per_frame)
    spiker_tid, peak = select_spiker_track(
        tracker,
        total_frames=total_frames,
        video_w=width,
        video_h=height,
        min_track_len=min_track_len,
    )

    return TrackedPoseRun(
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        poses_per_frame=poses_per_frame,
        track_ids_per_frame=track_ids_per_frame,
        spiker_track_id=spiker_tid,
        spiker_peak_velocity_px=peak,
    )


def extract_right_wrist_xy_from_run(
    run: TrackedPoseRun,
    *,
    visibility_thresh: float = DEFAULT_VISIBILITY_THRESH,
) -> np.ndarray:
    """Pick the spiker's right-wrist (x_px, y_px) for every frame of a tracked run."""
    xy = np.full((run.total_frames, 2), np.nan, dtype=np.float32)
    if run.spiker_track_id is None:
        return xy
    target = int(run.spiker_track_id)
    for f_idx in range(run.total_frames):
        for pose, tid in zip(run.poses_per_frame[f_idx], run.track_ids_per_frame[f_idx]):
            if tid != target:
                continue
            if len(pose) <= RIGHT_WRIST_INDEX:
                continue
            lm = pose[RIGHT_WRIST_INDEX]
            if lm[3] < visibility_thresh:
                continue
            xy[f_idx, 0] = lm[0] * run.width
            xy[f_idx, 1] = lm[1] * run.height
            break
    return xy


# ---------------------------------------------------------------------------
# Pose landmark extraction over a whole video
# ---------------------------------------------------------------------------
def extract_right_wrist_xy(
    video_path: str | Path,
    *,
    model_complexity: int = 1,
    model_dir: Path = DEFAULT_MODEL_DIR,
    num_poses: int = DEFAULT_NUM_POSES,
) -> np.ndarray:
    """
    Run MediaPipe Pose over every video frame and return the **spiker's** right-wrist
    pixel coordinates.

    Multiple poses (default ``num_poses=5``) are detected per frame and tracked with a
    nearest-neighbour mid-hip tracker; the track with the highest peak wrist velocity
    is selected as the spiker. Missing frames yield NaN rows (interpolated later by
    :func:`compute_kinematics`).

    Returns:
        Array of shape ``(n_frames, 2)`` with columns ``(x_px, y_px)``.
    """
    run = run_pose_tracking_on_video(
        video_path,
        model_complexity=model_complexity,
        model_dir=model_dir,
        num_poses=num_poses,
    )
    if run.spiker_track_id is None:
        print(
            f"[pose] no usable track in {Path(video_path).name} — wrist features will be all-NaN."
        )
    return extract_right_wrist_xy_from_run(run)


# ---------------------------------------------------------------------------
# Kinematics + NaN handling
# ---------------------------------------------------------------------------
def interpolate_nan_1d(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs, with forward/backward fill at the edges."""
    s = pd.Series(arr.astype(np.float64))
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.fillna(0.0)
    return s.to_numpy(dtype=np.float32)


def compute_kinematics(wrist_xy: np.ndarray) -> pd.DataFrame:
    """
    Turn a (N, 2) wrist-position array into a per-frame feature frame.

    Returns a DataFrame with columns:
      ``frame, wrist_x, wrist_y, wrist_velocity, wrist_acceleration``

    All features are NaN-safe (interpolated + filled).
    """
    if wrist_xy.ndim != 2 or wrist_xy.shape[1] != 2:
        raise ValueError(f"Expected (N,2) wrist_xy, got shape {wrist_xy.shape}")

    x = interpolate_nan_1d(wrist_xy[:, 0])
    y = interpolate_nan_1d(wrist_xy[:, 1])

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    velocity = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    acceleration = np.diff(velocity, prepend=velocity[0]).astype(np.float32)

    return pd.DataFrame(
        {
            "frame": np.arange(len(x), dtype=np.int64),
            "wrist_x": x,
            "wrist_y": y,
            "wrist_velocity": velocity,
            "wrist_acceleration": acceleration,
        }
    )


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------
def pose_cache_path(video_name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"{Path(video_name).stem}.csv"


def extract_pose_features_for_video(
    video_path: str | Path,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
    model_complexity: int = 1,
    model_dir: Path = DEFAULT_MODEL_DIR,
    num_poses: int = DEFAULT_NUM_POSES,
) -> pd.DataFrame:
    """Extract pose features for a single video with on-disk caching."""
    vp = Path(video_path)
    cache_file = pose_cache_path(vp.name, cache_dir)
    if cache_file.is_file() and not overwrite:
        return pd.read_csv(cache_file)

    wrist_xy = extract_right_wrist_xy(
        vp,
        model_complexity=model_complexity,
        model_dir=model_dir,
        num_poses=num_poses,
    )
    feats = compute_kinematics(wrist_xy)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cache_file, index=False)
    return feats


def batch_extract(
    video_names: Iterable[str],
    raw_dir: Path = DEFAULT_RAW_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    overwrite: bool = False,
    model_complexity: int = 1,
    model_dir: Path = DEFAULT_MODEL_DIR,
    num_poses: int = DEFAULT_NUM_POSES,
) -> dict[str, pd.DataFrame]:
    """Extract (or load from cache) pose features for many videos."""
    out: dict[str, pd.DataFrame] = {}
    for name in video_names:
        path = Path(raw_dir) / name
        if not path.is_file():
            print(f"[skip] missing video: {path}")
            continue
        print(f"[pose] {name} ...")
        out[name] = extract_pose_features_for_video(
            path,
            cache_dir=Path(cache_dir),
            overwrite=overwrite,
            model_complexity=model_complexity,
            model_dir=model_dir,
            num_poses=num_poses,
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Cache MediaPipe pose features per video.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Re-run extraction even if cached.")
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0=lite, 1=full, 2=heavy (Tasks API model variant).",
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        default=DEFAULT_NUM_POSES,
        help="Max poses MediaPipe returns per frame; the spiker is the track with peak wrist velocity.",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Optional list of video names (defaults to all .mp4 in --raw-dir).",
    )
    args = parser.parse_args()

    if args.videos:
        names = list(args.videos)
    else:
        names = sorted(p.name for p in Path(args.raw_dir).glob("*.mp4"))
        names += sorted(p.name for p in Path(args.raw_dir).glob("*.MP4"))

    if not names:
        print(f"No videos found in {args.raw_dir}")
        return

    batch_extract(
        names,
        raw_dir=Path(args.raw_dir),
        cache_dir=Path(args.cache_dir),
        overwrite=args.overwrite,
        model_complexity=args.model_complexity,
        model_dir=Path(args.model_dir),
        num_poses=args.num_poses,
    )
    print(f"Pose feature cache: {Path(args.cache_dir).resolve()}")


if __name__ == "__main__":
    main()

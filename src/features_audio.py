"""
Audio features per video frame using librosa.

For each video, we extract two signals and align them to video frame indices:

  * ``audio_rms``    — short-time RMS energy (peaks at the ball-hand impact)
  * ``audio_onset``  — onset strength (complementary to RMS; emphasizes transients)

Alignment: audio analysis runs on its own time base (``hop_length`` samples); we then
linearly interpolate those values onto each video frame's timestamp
``frame / fps``.

librosa cannot decode ``.mp4`` directly without ``ffmpeg`` available to ``audioread``
or ``soundfile``. To keep this self-contained we **demux the audio track to a temporary
WAV** using whichever FFmpeg we can find:

  1. ``ffmpeg`` on the system PATH (preferred), or
  2. the binary that ships with ``imageio-ffmpeg`` (``pip install imageio-ffmpeg``).

We then call ``librosa.load`` on the WAV and delete the temp file. This works on any
OS without requiring a system-wide ``apt install ffmpeg``.

Features are cached to ``data/processed/features/audio/<video_stem>.csv`` so the
extraction step only runs once per video.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Import sibling module when running as `python src/features_audio.py`
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from extract_frames import get_video_metadata  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "spike_clips"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "contact_frames.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "audio"

AUDIO_FEATURE_COLUMNS = ("audio_rms", "audio_onset")

DEFAULT_HOP_LENGTH = 512
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_AUDIO_SR = 22050  # librosa's default; sufficient for RMS / onset


# ---------------------------------------------------------------------------
# FFmpeg shim — demux video → mono WAV
# ---------------------------------------------------------------------------
def _resolve_ffmpeg_binary() -> str:
    """Return the path to a usable ``ffmpeg`` executable, or raise."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Could not find an ffmpeg binary. Install one of:\n"
            "  - system: 'sudo apt install ffmpeg'\n"
            "  - or pip: 'pip install imageio-ffmpeg' (bundles a self-contained ffmpeg)\n"
            f"Underlying error: {e}"
        )


def demux_audio_to_wav(video_path: str | Path, sample_rate: int = DEFAULT_AUDIO_SR) -> Path:
    """
    Extract the audio track of ``video_path`` to a mono WAV at ``sample_rate``.

    Returns the path to a temporary WAV file. The caller is responsible for deleting it.
    """
    ffmpeg = _resolve_ffmpeg_binary()
    fd, tmp_path = tempfile.mkstemp(prefix="cv_audio_", suffix=".wav")
    os.close(fd)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",  # no video
        "-ac",
        "1",  # mono
        "-ar",
        str(int(sample_rate)),
        "-loglevel",
        "error",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg failed to demux {video_path}: {e}")
    return Path(tmp_path)


# ---------------------------------------------------------------------------
# Core audio extraction
# ---------------------------------------------------------------------------
def compute_audio_arrays(
    video_path: str | Path,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    sample_rate: int = DEFAULT_AUDIO_SR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return RMS, onset strength, and the matching audio-frame time grid.

    Returns:
        rms:      (T,)
        onset:    (T,)  (zero-padded / trimmed to match rms length)
        times:    (T,)  seconds from start of the audio track
    """
    import librosa

    wav_path = demux_audio_to_wav(video_path, sample_rate=sample_rate)
    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # librosa can return slightly different T for rms vs onset — align to rms length.
    T = len(rms)
    if len(onset) < T:
        onset = np.pad(onset, (0, T - len(onset)), mode="constant")
    else:
        onset = onset[:T]

    times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)
    return rms.astype(np.float32), onset.astype(np.float32), times.astype(np.float64)


def align_audio_to_video_frames(
    rms: np.ndarray,
    onset: np.ndarray,
    audio_times: np.ndarray,
    fps: float,
    total_frames: int,
) -> pd.DataFrame:
    """
    Resample audio arrays onto video frame timestamps ``t = frame / fps``.

    Uses linear interpolation, with edge values held at the boundary (numpy's default).

    Returns DataFrame with columns ``frame, audio_rms, audio_onset``.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    frame_idx = np.arange(int(total_frames), dtype=np.int64)
    frame_times = frame_idx.astype(np.float64) / float(fps)

    rms_aligned = np.interp(frame_times, audio_times, rms).astype(np.float32)
    onset_aligned = np.interp(frame_times, audio_times, onset).astype(np.float32)

    return pd.DataFrame(
        {
            "frame": frame_idx,
            "audio_rms": rms_aligned,
            "audio_onset": onset_aligned,
        }
    )


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------
def audio_cache_path(video_name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"{Path(video_name).stem}.csv"


def extract_audio_features_for_video(
    video_path: str | Path,
    fps: float,
    total_frames: int,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
    hop_length: int = DEFAULT_HOP_LENGTH,
    frame_length: int = DEFAULT_FRAME_LENGTH,
) -> pd.DataFrame:
    """Extract audio features for one video with on-disk caching."""
    vp = Path(video_path)
    cache_file = audio_cache_path(vp.name, cache_dir)
    if cache_file.is_file() and not overwrite:
        return pd.read_csv(cache_file)

    rms, onset, times = compute_audio_arrays(
        vp, hop_length=hop_length, frame_length=frame_length
    )
    feats = align_audio_to_video_frames(rms, onset, times, fps=fps, total_frames=total_frames)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cache_file, index=False)
    return feats


def _list_mp4_in_raw_dir(raw_dir: Path) -> list[str]:
    """All ``.mp4`` / ``.MP4`` basenames in ``raw_dir``, sorted, unique."""
    seen: set[str] = set()
    for pat in ("*.mp4", "*.MP4"):
        for p in Path(raw_dir).glob(pat):
            seen.add(p.name)
    return sorted(seen)


def build_audio_jobs(
    raw_dir: Path,
    labels_df: pd.DataFrame | None,
    *,
    all_videos: bool = False,
    videos: list[str] | None = None,
) -> list[tuple[str, float, int]]:
    """
    Return ``(video_name, fps, total_frames)`` for each video to process.

    * ``--videos a.mp4 b.mp4`` — explicit list; metadata from each file.
    * ``--all-videos`` — every ``.mp4`` under ``raw_dir``; metadata from each file
      (same idea as :func:`features_pose.batch_extract` when no label list is used).
    * default — each row of ``contact_frames.csv`` (needs ``fps`` and ``total_frames``).

    New clips must either be **added to** ``contact_frames.csv`` *or* processed with
    ``--all-videos`` / ``--videos`` so audio extraction runs for them.
    """
    raw_dir = Path(raw_dir)
    if videos:
        out: list[tuple[str, float, int]] = []
        for name in videos:
            vp = raw_dir / name
            if not vp.is_file():
                print(f"[skip] missing video: {vp}")
                continue
            m = get_video_metadata(vp)
            if m.fps <= 0 or m.total_frames <= 0:
                print(f"[skip] bad metadata (fps={m.fps}, frames={m.total_frames}): {vp}")
                continue
            out.append((name, m.fps, m.total_frames))
        return out

    if all_videos:
        out = []
        for name in _list_mp4_in_raw_dir(raw_dir):
            vp = raw_dir / name
            m = get_video_metadata(vp)
            if m.fps <= 0 or m.total_frames <= 0:
                print(f"[skip] bad metadata (fps={m.fps}, frames={m.total_frames}): {vp}")
                continue
            out.append((name, m.fps, m.total_frames))
        return out

    if labels_df is None or len(labels_df) == 0:
        raise ValueError(
            "No labels to process. Use --all-videos to scan --raw-dir for .mp4 files, "
            "or --videos to name specific files, or pass a non-empty contact_frames.csv."
        )
    required = {"video_name", "fps", "total_frames"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"Labels DataFrame missing columns: {missing}")

    out2: list[tuple[str, float, int]] = []
    for _, row in labels_df.iterrows():
        name = str(row["video_name"])
        out2.append((name, float(row["fps"]), int(row["total_frames"])))
    return out2


def batch_extract(
    labels_df: pd.DataFrame | None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    overwrite: bool = False,
    all_videos: bool = False,
    videos: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Cache audio features for each target video.

    See :func:`build_audio_jobs` for how the target list is built.
    """
    jobs = build_audio_jobs(
        raw_dir, labels_df, all_videos=all_videos, videos=videos
    )
    out: dict[str, pd.DataFrame] = {}
    for name, fps, total_frames in jobs:
        vp = Path(raw_dir) / name
        if not vp.is_file():
            print(f"[skip] missing video: {vp}")
            continue
        print(f"[audio] {name} ...")
        out[name] = extract_audio_features_for_video(
            vp,
            fps=float(fps),
            total_frames=int(total_frames),
            cache_dir=Path(cache_dir),
            overwrite=overwrite,
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Cache per-frame audio features (RMS + onset).")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Re-run extraction even if cached.")
    parser.add_argument(
        "--all-videos",
        action="store_true",
        help="Process every .mp4 under --raw-dir; read fps and total_frames from each file. "
        "Use this for new clips that are not yet in contact_frames.csv (matches features_pose.py behavior).",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Optional explicit list of video basenames; metadata read from file.",
    )
    args = parser.parse_args()

    labels_df: pd.DataFrame | None = None
    if not args.all_videos and not args.videos:
        if not args.labels.is_file():
            raise FileNotFoundError(
                f"Labels CSV not found: {args.labels}\n"
                "Either add the file, or use --all-videos / --videos to process clips without it."
            )
        labels_df = pd.read_csv(args.labels)

    batch_extract(
        labels_df,
        raw_dir=Path(args.raw_dir),
        cache_dir=Path(args.cache_dir),
        overwrite=args.overwrite,
        all_videos=bool(args.all_videos),
        videos=list(args.videos) if args.videos else None,
    )
    print(f"Audio feature cache: {Path(args.cache_dir).resolve()}")


if __name__ == "__main__":
    main()

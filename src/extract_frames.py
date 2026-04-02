"""
Extract frames from volleyball (or other) videos using OpenCV.

Use this module to:
  - Read video metadata (fps, frame count, size).
  - Iterate frames without loading the whole video into RAM.
  - Save every frame to disk as an image, or collect frames in memory (smaller clips only).

Frame indices are **0-based**: first frame is index 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2

# Default folder layout (relative to project root = parent of `src/`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"


@dataclass
class VideoMetadata:
    """Basic metadata for one video file."""

    video_path: Path
    video_name: str
    fps: float
    total_frames: int
    width: int
    height: int


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """
    Open a video and read metadata without decoding every pixel.

    Args:
        video_path: Path to a video file (e.g. .mp4).

    Returns:
        VideoMetadata with fps, frame count, and resolution.

    Note:
        OpenCV's reported frame count can occasionally be off by 1 on some codecs;
        for ground truth, prefer counting frames while reading if you need exactness.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return VideoMetadata(
        video_path=path.resolve(),
        video_name=path.name,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
    )


def iter_frames(video_path: str | Path) -> Iterator[tuple[int, Any]]:
    """
    Yield (frame_index, frame_bgr) for each frame in the video.

    Frames are in **BGR** uint8 layout, as returned by OpenCV.

    This is memory-friendly: only one frame exists in memory at a time.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def extract_frames_to_memory(video_path: str | Path) -> tuple[VideoMetadata, list[Any]]:
    """
    Read **all** frames into a Python list (BGR images).

    Warning:
        Large videos can use a lot of RAM. Prefer ``iter_frames`` or
        ``extract_frames_to_disk`` for long clips.

    Returns:
        (metadata, frames) where frames[i] is the i-th frame (0-based).
    """
    meta = get_video_metadata(video_path)
    frames: list[Any] = []
    for i, frame in iter_frames(video_path):
        frames.append(frame)
    # Prefer actual count from decoding (handles quirky CAP_PROP_FRAME_COUNT)
    actual = len(frames)
    if actual != meta.total_frames:
        meta = VideoMetadata(
            video_path=meta.video_path,
            video_name=meta.video_name,
            fps=meta.fps,
            total_frames=actual,
            width=meta.width,
            height=meta.height,
        )
    return meta, frames


def extract_frames_to_disk(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    image_ext: str = ".jpg",
    jpeg_quality: int = 95,
) -> VideoMetadata:
    """
    Decode every frame and save it under ``output_dir``.

    Files are named with zero-padded indices: ``000000.jpg``, ``000001.jpg``, ...

    Args:
        video_path: Input video.
        output_dir: Folder to create (e.g. ``data/processed/frames/clip_01``).
        image_ext: ``.jpg`` or ``.png`` (default .jpg for size).
        jpeg_quality: 0-100 for JPEG compression.

    Returns:
        VideoMetadata (total_frames updated to the number of frames actually written).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = Path(video_path)
    meta0 = get_video_metadata(path)
    count = 0

    for idx, frame in iter_frames(path):
        fname = f"{idx:06d}{image_ext}"
        fpath = out / fname
        if image_ext.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(
                str(fpath),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
        else:
            cv2.imwrite(str(fpath), frame)
        count += 1

    return VideoMetadata(
        video_path=meta0.video_path,
        video_name=meta0.video_name,
        fps=meta0.fps,
        total_frames=count,
        width=meta0.width,
        height=meta0.height,
    )


def extract_frames(
    video_path: str | Path,
    *,
    save_dir: str | Path | None = None,
    return_arrays: bool = True,
    image_ext: str = ".jpg",
    jpeg_quality: int = 95,
) -> tuple[VideoMetadata, list[Any] | None]:
    """
    Extract all frames from one video — either save to disk, load into memory, or both.

    Args:
        video_path: Path to the video.
        save_dir: If set, each frame is written under this directory.
        return_arrays: If True, also return a list of all frames (None if False).
        image_ext / jpeg_quality: Passed to ``extract_frames_to_disk`` when saving.

    Returns:
        (metadata, frames_or_none)
        ``frames_or_none`` is a list of BGR arrays when ``return_arrays`` is True,
        otherwise None (e.g. when you only want files on disk).
    """
    path = Path(video_path)
    frames: list[Any] | None = [] if return_arrays else None

    if save_dir is not None:
        meta_saved = extract_frames_to_disk(
            path, save_dir, image_ext=image_ext, jpeg_quality=jpeg_quality
        )
        if not return_arrays:
            return meta_saved, None
        # Need arrays: read back from disk could be slow; decode once from video instead
        _, mem_frames = extract_frames_to_memory(path)
        meta = VideoMetadata(
            video_path=meta_saved.video_path,
            video_name=meta_saved.video_name,
            fps=meta_saved.fps,
            total_frames=len(mem_frames),
            width=meta_saved.width,
            height=meta_saved.height,
        )
        return meta, mem_frames

    meta, mem_frames = extract_frames_to_memory(path)
    return meta, mem_frames


def list_videos(raw_dir: str | Path | None = None, extensions: tuple[str, ...] = (".mp4", ".avi", ".mov")) -> list[Path]:
    """Return sorted video paths under ``raw_dir`` (default: ``data/raw_videos``)."""
    d = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_VIDEOS_DIR
    if not d.is_dir():
        return []
    out: list[Path] = []
    for ext in extensions:
        out.extend(d.glob(f"*{ext}"))
        out.extend(d.glob(f"*{ext.upper()}"))
    return sorted({p.resolve() for p in out})


def main() -> None:
    """
    CLI demo: list videos in ``data/raw_videos``, print metadata, optionally save frames.

    By default this only **prints metadata** (does not dump thousands of images).
    Set environment variable or edit SAVE_FRAMES below to True to write frames to
    ``data/processed/frames/<video_stem>/``.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from raw videos (OpenCV).")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_VIDEOS_DIR,
        help="Directory containing raw videos.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, save all frames under data/processed/frames/<stem>/",
    )
    parser.add_argument(
        "--processed-frames-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "frames",
        help="Root folder for saved frame sequences.",
    )
    args = parser.parse_args()

    videos = list_videos(args.raw_dir)
    if not videos:
        print(f"No videos found in {args.raw_dir}")
        return

    for vp in videos:
        meta = get_video_metadata(vp)
        print(f"{meta.video_name}: fps={meta.fps:.3f}, frames={meta.total_frames}, size={meta.width}x{meta.height}")

        if args.save:
            out_dir = args.processed_frames_root / Path(vp).stem
            meta2 = extract_frames_to_disk(vp, out_dir)
            print(f"  saved {meta2.total_frames} frames -> {out_dir}")


if __name__ == "__main__":
    main()

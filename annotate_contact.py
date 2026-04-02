import cv2
import os
import csv

VIDEO_DIR = "data/spike_clips"
OUTPUT_CSV = "data/labels/contact_frames.csv"

WINDOW_NAME = "Contact Annotation"
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps, total_frames


def draw_frame(frame, frame_idx, total_frames, video_name):
    display = frame.copy()
    display = cv2.resize(display, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    text_1 = f"Video: {video_name}"
    text_2 = f"Frame: {frame_idx}/{total_frames - 1}"
    text_3 = "a: prev  d: next  s: save  q: quit"

    cv2.putText(display, text_1, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, text_2, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, text_3, (20, DISPLAY_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return display


def annotate_video(video_path):
    video_name = os.path.basename(video_path)
    frames, fps, total_frames = load_video_frames(video_path)

    if len(frames) == 0:
        print(f"Could not read frames from {video_name}")
        return None, fps, total_frames

    frame_idx = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, 100, 80)

    while True:
        display = draw_frame(frames[frame_idx], frame_idx, total_frames, video_name)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("d"):  # next frame
            frame_idx = min(frame_idx + 1, total_frames - 1)

        elif key == ord("a"):  # previous frame
            frame_idx = max(frame_idx - 1, 0)

        elif key == ord("s"):  # save contact frame
            print(f"Saved {video_name}: contact_frame={frame_idx}, fps={fps}, total_frames={total_frames}")
            cv2.destroyWindow(WINDOW_NAME)
            return frame_idx, fps, total_frames

        elif key == ord("q"):  # quit without saving
            print(f"Skipped {video_name}")
            cv2.destroyWindow(WINDOW_NAME)
            return None, fps, total_frames


def get_video_files(video_dir):
    valid_exts = (".mp4", ".mov", ".avi", ".mkv")
    return sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(valid_exts)
    ])


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    video_files = get_video_files(VIDEO_DIR)

    if not video_files:
        print(f"No video files found in {VIDEO_DIR}")
        return

    rows = []
    for video_name in video_files:
        video_path = os.path.join(VIDEO_DIR, video_name)
        contact_frame, fps, total_frames = annotate_video(video_path)

        if contact_frame is not None:
            rows.append([video_name, contact_frame, fps, total_frames])

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "contact_frame", "fps", "total_frames"])
        writer.writerows(rows)

    print(f"\nSaved annotations to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
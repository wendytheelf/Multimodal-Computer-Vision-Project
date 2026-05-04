# Multimodal Volleyball Spike Contact Timing Analysis

Author: Wendy Hu  

---

# 1. Problem Framing + Scope

## Project Goal

The goal of this project is to automatically detect the **contact moment of a volleyball spike** using multimodal signals from video and audio.

Instead of classifying whether a spike is "good" or "bad", this project focuses on **biomechanical timing analysis**.

The system detects:

- Contact frame (ball-hand impact)
- Jump apex frame (highest point of jump)
- Timing offset between these two events

This helps determine whether the spike contact occurs **too early, too late, or near the optimal moment**.

---

# 2. Task Definition

Input:

volleyball spike video clip

Output:

contact_frame
jump_apex_frame
timing_offset

Example Output:

Contact frame: 32
Jump apex frame: 35
Timing offset: -3 frames
Interpretation: Early contact

---

# 3. Dataset Access + Documentation

## Data Source

The dataset consists of volleyball spike clips collected from practice videos.

Current dataset:

- spike clips
- Duration: 2–5 seconds
- Estimated FPS: ~30

Each clip contains:

- Approach
- Jump
- Arm swing
- Ball contact
- Landing

---

## Data Structure

data/
│
├── spike_clips/
│ ├── V3.mp4
│ ├── V4.mp4
│
├── annotations/
│ ├── contact_labels.csv


Videos are not included in the repository due to size.

---

# 4. System Architecture

The pipeline of the project is:

Video Clip
↓
YOLO Person Detection
↓
Player Localization
↓
Pose Estimation
↓
Keypoint Time Series
↓
Pose-based Contact Detection
↓
Audio Extraction
↓
Audio-based Contact Detection
↓
Multimodal Fusion
↓
Jump Apex Detection
↓
Timing Analysis


---

# 5. Object Detection

We use **YOLOv8** to detect the player in each frame.

Output:

person bounding box


The bounding box containing the main player is used to crop the region of interest.

---

# 6. Pose Estimation

Pose estimation extracts body keypoints from the cropped player region.

Model used:

MediaPipe Pose

Output:

33 body keypoints (x, y coordinates)


These keypoints form a time-series representation of the spike motion.

---

# 7. Pose-Based Contact Detection

Observation:

The wrist velocity typically peaks at the moment of ball contact.

Baseline detection:

contact_frame = argmax(wrist_velocity)


---

# 8. Audio-Based Contact Detection

The impact between the hand and the ball produces a short, high-energy sound.

Audio processing steps:

1. Extract waveform
2. Compute short-time energy
3. Detect energy peak

Output:

t_audio_contact

---

# 9. Multimodal Fusion

The final contact estimate combines pose-based and audio-based signals.

Fusion rule:

if |t_pose - t_audio| < threshold:
fused = average
else:
fused = audio

---

# 10. Jump Apex Detection

Jump apex is estimated using hip trajectory.

Method:

apex_frame = argmin(hip_y)


This corresponds to the highest point of the jump.

---

# 11. Evaluation Plan

Ground truth:

Manual annotation of contact frames.

Metrics:

- Mean Absolute Error (MAE)
- Frame error
- Accuracy within ±2 frames

Comparison methods:

1. Pose-only detection
2. Audio-only detection
3. Multimodal fusion

---

# 12. Check-in 1 Progress

Completed:

- Dataset collection (spike clips)
- Initial pose extraction experiments
- Exploratory data analysis
- Baseline contact detection

EDA includes:

- Pose visualization
- Wrist velocity analysis
- Audio energy visualization

---

# 13. Next Steps

Future work includes:

- Implement YOLO player detection
- Expand dataset
- Implement audio-based detection
- Multimodal fusion
- Full evaluation

---

# 14. Project Significance

This project combines:

- Object detection (YOLO)
- Pose estimation
- Audio signal analysis
- Multimodal fusion

to study biomechanical timing in volleyball spikes.


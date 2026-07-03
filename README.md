# Multi-Tiered Computational Oculometrics: Webcam-Based Gaze Tracking via ResNet-50 for Neurodevelopmental Screening Baselines

An undergraduate dissertation exploring the engineering constraints, signal-to-noise ratio (SNR) thresholds, and spatial optimization matrices of consumer-grade webcam gaze-tracking architectures. This pipeline maps continuous 3D gaze vector regressions onto a 2D screen coordinate system to establish baseline diagnostics suitable for accessible digital health triage.

📌 Project Overview
Traditional oculomotor diagnostic platforms require dedicated high-frequency (300Hz–600Hz) infrared hardware. This project evaluates a low-cost machine learning alternative: leveraging a standard 30Hz consumer webcam combined with a Deep Convolutional Neural Network (ResNet-50) and MediaPipe face mesh extraction to regress and scale real-time gaze metrics.

### Key Research Questions Addressed
1. **The Hardware-Fidelity Constraint:** Can a standard 30Hz 1080p webcam capture macroscopic saccadic movements with adequate accuracy despite low temporal resolution?
2. **The Spatial SNR Trade-off:** How do physical facial boundaries, eyelid occlusions, and head movements affect the Horizontal (Yaw) vs. Vertical (Pitch) signal-to-noise ratios?
3. **Dynamic Alignment:** Can linear extrapolation frameworks effectively map bounded eye-rotation samples to widescreen displays without triggering cervical (neck) rotation artifacts?

---

## 🛠️ Architecture & Pipeline Processing Flow

[Webcam Input: 1080p @ 30Hz]
│
▼
[MediaPipe Face Detection] ──► Extracted Regions of Interest (ROI) & Dynamic Centroid
│
▼
[Image Preprocessing] ──────► 224x224 Resize, Tensor Normalization
│
▼
[ResNet-50 Deep Network] ───► Continuous Multi-Class Linear Layer
│
▼
[Raw Gaze Estimation] ─────► 3D Gaze Vectors (Yaw / Pitch Coefficients)
│
▼
[Calibration Mapping] ──────► JSON Spatial Extrapolation Matrix & Moving-Head Offset Fix
│
▼
[Temporal Queue Smoothing] ──► 7-Frame Moving Average Window
│
▼
[Screen Target Output] ────► Real-Time Coordinates Displayed on Screen

----

## 📈 Key Findings & Dissertation Discussions

### 1. The Signal-to-Noise Ratio (SNR) Dilemma
A core engineering outcome of this study was tracking the relationship between **Gaze Signal** and **Webcam Ambient Noise**. When looking from edge to edge on a widescreen monitor, a user's pupils move a physical distance that translates to a highly restricted range in raw AI coefficient space (~100 arbitrary units). Stretching this small range across a **1920x1080 canvas** requires a high sensitivity multiplier, meaning minor illumination flickers or micro-head movements are drastically amplified. 

### 2. Horizontal (Yaw) vs. Vertical (Pitch) Variance
Empirical testing revealed a profound asymmetrical accuracy split between tracking coordinates:
* **Horizontal Axes (Yaw):** Maintained high robust structural accuracy. Pupil displacement against the white sclera provides high-contrast edge gradients that the ResNet-50 can easily isolate.
* **Vertical Axes (Pitch):** Subject to a severe drop in SNR. Downward gaze angles naturally introduce eyelid drooping (ptosis), obscuring the upper iris and causing a compression of the spatial signal. 

---

## 🚀 Directory Structure & Modules

* `train_gaze_4090.py`: Script optimized for GPU-accelerated (RTX 4090) fine-tuning of the ResNet-50 network architecture.
* `calibration.py`: Interactive spacebar-capture tool implementing a central bounding box to sample metrics without head-rotation bias.
* `test_my_model.py`: Main live deployment engine combining MediaPipe face tracking, ResNet inference, and moving-head-shift compensation.
* `calibration.json`: Config file output storing real-time scaling offsets, raw sensitivity values, and positional anchor coordinates.
* `visualise_error.py` / `analyze_results.py`: Verification toolkits calculating root-mean-square errors (RMSE) and generating analytical validation figures.

---

## 💻 Getting Started & Installation

### 1. Prerequisites
Ensure you have a Python environment (3.10+) running with PyTorch configured for CUDA execution if using GPU training:

pip install torch torchvision numpy opencv-python mediapipe timm pillow

### 2. Running Calibration

To establish your personal eye-movement boundaries without moving your neck, run the target collector script:

python calibration.py

Look directly at the inner targets as they appear on the screen and press SPACEBAR to snap coordinates while holding your head entirely still.

### 3. Launching Live Tracker

Once your calibration.json file is populated, initialize the runtime system:

python test_my_model.py

Press q to terminate the live frame sequence window.

"""
THREE-STAGE GAZE SYSTEM TEST SCRIPT
Proves: AI stability (Stage 1), Hardware noise (Stage 2), Filter effectiveness (Stage 3)

Stage 1: Digital Benchmark - Static image (proves AI is stable)
Stage 2: Sensor Benchmark - Webcam with printed photo (proves hardware is noisy)  
Stage 3: Physiological Benchmark - You as subject (proves filter works with human noise)

Set TEST_STAGE and INPUT_MODE below, then run.
"""

import torch
import cv2
import numpy as np
import mediapipe as mp
import json
import os
from timm import create_model
from torchvision import transforms
from PIL import Image
from collections import deque
from statistics import median
import warnings
from datetime import datetime
import time
import glob


# =============================================================================
# TEST CONFIGURATION - SET THESE BEFORE RUNNING
# =============================================================================
TEST_STAGE = 4
INPUT_MODE = "WEBCAM"  # "IMAGE" for stage 1, "WEBCAM" for stages 2, 3, and 4
USE_FILTER = True      # Used to toggle the EMA/deadzone filter (False for Stage 3)
IMAGE_PATH = "dataset/Data/Normalized/p00/p00-Normalized-0.jpg"  # Path to test image for Stage 1

# Override for manual testing
# INPUT_MODE = "IMAGE"  # Switch this to WEBCAM to test with live camera


def get_latest_file(pattern, directory="."):
    """Find the most recent file matching a pattern."""
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# =============================================================================
# RESOLUTION CONSTANTS - 4K Display Configuration
# =============================================================================
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160
WEBCAM_WIDTH = 1920
WEBCAM_HEIGHT = 1080

Y_LOCK_POSITION = SCREEN_HEIGHT // 2
FIXATION_FRAMES = 300
FIXATION_TARGET_X = SCREEN_WIDTH // 2
FIXATION_TARGET_Y = SCREEN_HEIGHT // 2
ROSE_SNR_THRESHOLD = 5.0
NUM_SAFE_ZONES = 3


# =============================================================================
# SNR & PRECISION ANALYSIS FUNCTIONS
# =============================================================================
def calculate_snr(signal_range, noise_std):
    """Calculate Signal-to-Noise Ratio using Rose Criterion (1948)."""
    if noise_std == 0:
        return float('inf')
    return signal_range / noise_std


def analyze_fixation_data(log_data):
    """Perform comprehensive SNR and precision analysis on fixation data."""
    if len(log_data) < 10:
        return {"error": "Insufficient data for analysis"}

    mapped_x = np.array([entry['mapped_x'] for entry in log_data])
    mapped_y = np.array([entry['mapped_y'] for entry in log_data])
    raw_yaw = np.array([entry['raw_yaw'] for entry in log_data])
    raw_pitch = np.array([entry['raw_pitch'] for entry in log_data])

    sigma_x = np.std(mapped_x)
    sigma_y = np.std(mapped_y)
    sigma_raw_yaw = np.std(raw_yaw)
    sigma_raw_pitch = np.std(raw_pitch)

    snr_x = calculate_snr(SCREEN_WIDTH, sigma_x)
    snr_y = calculate_snr(SCREEN_HEIGHT, sigma_y)

    x_stable = bool(snr_x >= ROSE_SNR_THRESHOLD)
    y_stable = bool(snr_y >= ROSE_SNR_THRESHOLD)

    jitter_x = np.mean(np.abs(np.diff(mapped_x)))
    jitter_y = np.mean(np.abs(np.diff(mapped_y)))

    analysis = {
        "num_frames": len(log_data),
        "horizontal": {
            "mean": float(np.mean(mapped_x)),
            "std_sigma": float(sigma_x),
            "snr": float(snr_x),
            "stable": x_stable,
            "status": "STABLE" if x_stable else "MATHEMATICALLY UNSTABLE",
            "jitter_mean": float(jitter_x),
            "range": [float(np.min(mapped_x)), float(np.max(mapped_x))]
        },
        "vertical": {
            "mean": float(np.mean(mapped_y)),
            "std_sigma": float(sigma_y),
            "snr": float(snr_y),
            "stable": y_stable,
            "status": "STABLE" if y_stable else "MATHEMATICALLY UNSTABLE",
            "jitter_mean": float(jitter_y),
            "range": [float(np.min(mapped_y)), float(np.max(mapped_y))]
        },
        "raw_model_output": {
            "yaw_std": float(sigma_raw_yaw),
            "pitch_std": float(sigma_raw_pitch)
        },
        "rose_threshold": ROSE_SNR_THRESHOLD
    }

    return analysis


def calculate_safe_zones(sigma_x, num_zones=NUM_SAFE_ZONES):
    """Calculate 1D horizontal safe zones with 6-sigma guard bands."""
    zone_width = SCREEN_WIDTH / num_zones
    guard_band = 6 * sigma_x

    zones = []
    zone_names = ["LEFT", "CENTER", "RIGHT"] if num_zones == 3 else [f"ZONE_{i}" for i in range(num_zones)]

    for i in range(num_zones):
        zone_start = i * zone_width
        zone_end = (i + 1) * zone_width

        safe_start = zone_start + guard_band if i > 0 else zone_start
        safe_end = zone_end - guard_band if i < num_zones - 1 else zone_end

        zones.append({
            "name": zone_names[i] if i < len(zone_names) else f"ZONE_{i}",
            "raw_boundary": [zone_start, zone_end],
            "safe_boundary": [max(0, safe_start), min(SCREEN_WIDTH, safe_end)],
            "guard_band_px": guard_band
        })

    return zones


def get_current_zone(x_position, zones):
    """Determine which safe zone the current X position falls into."""
    for zone in zones:
        raw_start, raw_end = zone['raw_boundary']
        safe_start, safe_end = zone['safe_boundary']

        if raw_start <= x_position < raw_end:
            if safe_start <= x_position < safe_end:
                return zone['name'], "HIGH_CONFIDENCE"
            else:
                return zone['name'], "GUARD_BAND"

    return "OUT_OF_BOUNDS", "ERROR"


# =============================================================================
# SETUP & INITIALIZATION
# =============================================================================

print("\n" + "=" * 70)
print(f"THREE-STAGE TESTING - STAGE {TEST_STAGE}")
print(f"Input Mode: {INPUT_MODE}")
print("=" * 70 + "\n")

# 1. Load Calibration Data
calib_file = get_latest_file("calibration*.json")
if not calib_file:
    if os.path.exists("calibration.json"):
        calib_file = "calibration.json"
    else:
        print("ERROR: No calibration file found!")
        exit()

with open(calib_file, "r") as f:
    calib = json.load(f)
    print(f"✓ Calibration loaded from: {calib_file}")
    print(f"  sens_x: {calib['sens_x']:.4f}, off_x: {calib['off_x']:.4f}")

# 2. Setup Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('resnet50', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("gaze_resnet50_epoch10.pth", weights_only=True))
model.to(device)
model.eval()
print(f"✓ Model loaded on: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"gaze_log_{timestamp}.json"

# 4. Setup Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# 5. Setup Input Source (IMAGE or WEBCAM)
if INPUT_MODE == "IMAGE":
    print(f"✓ IMAGE MODE - Loading static image: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        exit()
    static_image = cv2.imread(IMAGE_PATH)
    if static_image is None:
        print(f"ERROR: Could not load image from {IMAGE_PATH}")
        exit()
    static_image = cv2.flip(static_image, 1)  # Flip for consistency
    h, w, _ = static_image.shape
    print(f"  Image dimensions: {w}x{h}")
    frame_source = "static_image"
else:
    print("✓ WEBCAM MODE - Using live webcam input")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Warm up camera
    for _ in range(3):
        cap.read()
    frame_source = "webcam"

window_name = f"TEST STAGE {TEST_STAGE} - {INPUT_MODE} MODE"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

# 6. Smoothing & Initialization
session_bias_x = 0
x_buffer = []
ema_x = 1920
alpha = 0.05
show_comparison = True
last_stable_x = 1920
deadzone_threshold = 50
history_y = deque(maxlen=7)
bbox_history_x = deque(maxlen=5)
bbox_history_y = deque(maxlen=5)
bbox_history_w = deque(maxlen=5)
bbox_history_h = deque(maxlen=5)

# 7. Test State
fixation_log = []
fixation_frame_count = 0
test_complete = False

print(f"\n" + "="*70)
print(f"STAGE {TEST_STAGE} TEST RUNNING")
print(f"Input: {INPUT_MODE} MODE")
print(f"Image path: {IMAGE_PATH if INPUT_MODE == 'IMAGE' else 'N/A (using webcam)'}")
print(f"{'=' * 70}\n")

# =============================================================================
# MAIN TEST LOOP
# =============================================================================
frame_count = 0
start_time = time.time()
collecting_data = False

while True:
    # Read frame from source
    if INPUT_MODE == "IMAGE":
        frame = static_image.copy()  # Use same static image every frame
    else:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    scale_x = SCREEN_WIDTH / w
    scale_y = SCREEN_HEIGHT / h

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    # Default values
    target_x = SCREEN_WIDTH // 2
    target_y = Y_LOCK_POSITION
    raw_yaw = 0.0
    raw_pitch = 0.0
    mapped_y_raw = Y_LOCK_POSITION
    raw_x_unbiased = 0

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box

            bbox_history_x.append(int(bboxC.xmin * w))
            bbox_history_y.append(int(bboxC.ymin * h))
            bbox_history_w.append(int(bboxC.width * w))
            bbox_history_h.append(int(bboxC.height * h))

            xmin = int(sum(bbox_history_x) / len(bbox_history_x))
            ymin = int(sum(bbox_history_y) / len(bbox_history_y))
            width = int(sum(bbox_history_w) / len(bbox_history_w))
            height = int(sum(bbox_history_h) / len(bbox_history_h))

            pad_x, pad_y = int(width * 0.2), int(height * 0.2)
            x1, y1 = max(0, xmin - pad_x), max(0, ymin - pad_y)
            x2, y2 = min(w, xmin + width + pad_x), min(h, ymin + height + pad_y)

            face_center_x = x1 + (x2 - x1) // 2
            face_center_y = y1 + (y2 - y1) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size != 0:
                input_tensor = transform(Image.fromarray(face_crop)).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    raw_pitch = output[0][0].item() * 1000.0
                    pred_yaw = output[0][1].item() * 1000.0
                    raw_yaw = pred_yaw

                # 1. RAW COORDINATE
                raw_x_unbiased = (pred_yaw * calib['sens_x']) + calib['off_x']
                
                # 2. AUTO-ZEROING BIAS
                raw_x_corrected = raw_x_unbiased + session_bias_x
                
                if USE_FILTER:
                    # 3. MEDIAN FILTERING
                    x_buffer.append(raw_x_corrected)
                    if len(x_buffer) > 5:
                        x_buffer.pop(0)
                    median_x = median(x_buffer)
                    
                    # 4. EMA SMOOTHING
                    ema_x = (alpha * median_x) + ((1 - alpha) * ema_x)

                    # 5. DEADZONE LOGIC
                    movement_velocity = abs(ema_x - last_stable_x)
                    if movement_velocity > deadzone_threshold:
                        final_display_x = ema_x
                        last_stable_x = ema_x
                    else:
                        final_display_x = last_stable_x
                else:
                    final_display_x = raw_x_corrected

                # 6. LOCK Y-AXIS
                target_y = Y_LOCK_POSITION
                target_x = int(final_display_x)
                target_x = max(0, min(SCREEN_WIDTH, target_x))

                mapped_y_raw = int((raw_pitch * calib['sens_y']) + calib['off_y'])
                mapped_y_raw = max(0, min(SCREEN_HEIGHT, mapped_y_raw))

                # Visual feedback
                if show_comparison:
                    cv2.circle(frame, (int(raw_x_corrected / scale_x), int(target_y / scale_y)), 
                               10, (0, 0, 255), 2)
                    cv2.putText(frame, "RAW (Flicker)", (int(raw_x_corrected / scale_x), int(target_y / scale_y) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.circle(frame, (int(final_display_x / scale_x), int(target_y / scale_y)), 
                           25, (0, 255, 0), -1)

    # Convert to display coordinates
    display_x = int(target_x / scale_x)
    display_y = int(target_y / scale_y)

    # Draw gaze point
    cv2.circle(frame, (display_x, display_y), 20, (0, 255, 0), -1)
    cv2.circle(frame, (display_x, display_y), 10, (255, 255, 255), -1)

    # Data collection mode
    if collecting_data:
        fixation_log.append({
            "frame": fixation_frame_count,
            "timestamp": time.time(),
            "raw_yaw": raw_yaw,
            "raw_pitch": raw_pitch,
            "mapped_x": target_x,
            "mapped_y": mapped_y_raw
        })
        fixation_frame_count += 1

        # Progress bar
        progress = fixation_frame_count / FIXATION_FRAMES
        bar_width = int(w * 0.6)
        bar_x = int(w * 0.2)
        cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_width, 50), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, 30), (bar_x + int(bar_width * progress), 50), (0, 255, 0), -1)
        cv2.putText(frame, f"COLLECTING: {fixation_frame_count}/{FIXATION_FRAMES}",
                    (bar_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check if test complete
        if fixation_frame_count >= FIXATION_FRAMES:
            collecting_data = False
            test_complete = True
            print("\n✓ Data collection complete! Analyzing...\n")

    # UI overlay
    cv2.putText(frame, f"STAGE {TEST_STAGE} - {INPUT_MODE}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if not collecting_data and not test_complete:
        cv2.putText(frame, "Press 'S' to START TEST, 'Q' to QUIT",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow(window_name, frame)
    frame_count += 1
    
    if frame_count == 1:
        print(f"✓ First frame displayed, waiting for 'S' key...")
        print(f"✓ Window: {window_name}")

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s') and not collecting_data and not test_complete:
        collecting_data = True
        fixation_log = []
        fixation_frame_count = 0
        
        # Auto-center gaze to prevent off-screen clamping
        session_bias_x = (SCREEN_WIDTH // 2) - raw_x_unbiased
        
        # INSTANTLY snap the exponential filters to the center so it doesn't "glide" on-screen
        ema_x = SCREEN_WIDTH // 2
        last_stable_x = SCREEN_WIDTH // 2
        x_buffer = [SCREEN_WIDTH // 2]
        
        print(f"► Starting test... collecting {FIXATION_FRAMES} frames (~10 seconds at 30fps)")

    if test_complete:
        break

cap.release() if INPUT_MODE == "WEBCAM" else None
cv2.destroyAllWindows()

# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================

if fixation_log:
    print("\n" + "=" * 70)
    print("SNR & PRECISION ANALYSIS REPORT (Rose Criterion 1948)")
    print("=" * 70)
    
    analysis = analyze_fixation_data(fixation_log)
    sigma_x = analysis['horizontal']['std_sigma']
    
    print(f"\nFrames Analyzed: {analysis['num_frames']}")
    print(f"Rose SNR Threshold: {ROSE_SNR_THRESHOLD}")
    print("-" * 70)

    print("\nHORIZONTAL AXIS (X) - Primary Tracking Dimension:")
    h_data = analysis['horizontal']
    print(f"  Mean: {h_data['mean']:.1f} px")
    print(f"  Std Dev (σ): {h_data['std_sigma']:.2f} px")
    print(f"  SNR (Signal/Noise): {h_data['snr']:.2f}")
    print(f"  Status: {h_data['status']}")
    print(f"  Range: [{h_data['range'][0]:.0f}, {h_data['range'][1]:.0f}] px")
    print(f"  Frame-to-Frame Jitter: {h_data['jitter_mean']:.2f} px")

    print("\nVERTICAL AXIS (Y) - 1D Pivot (Locked at center):")
    v_data = analysis['vertical']
    print(f"  Mean: {v_data['mean']:.1f} px (target: {Y_LOCK_POSITION})")
    print(f"  Std Dev (σ): {v_data['std_sigma']:.2f} px")
    print(f"  SNR (Signal/Noise): {v_data['snr']:.2f}")
    print(f"  Status: {v_data['status']}")

    print("\n" + "-" * 70)
    print("EXPECTED RESULTS BY STAGE:")
    print("-" * 70)
    if TEST_STAGE == 1:
        print("STAGE 1 (Digital Benchmark - Static Image):")
        print("  ✓ Expected: Jitter << 2 px, SNR >> 5.0")
        print("  ✓ Proves: AI model is completely stable")
        print(f"  ✓ Actual Jitter: {h_data['jitter_mean']:.2f} px")
        print(f"  ✓ Actual SNR: {h_data['snr']:.2f}")
    elif TEST_STAGE == 2:
        print("STAGE 2 (Sensor Benchmark - Printed Photo):")
        print("  ✓ Expected: Jitter 15-20 px, SNR drops significantly")
        print("  ✓ Proves: Hardware (1080p webcam → 4K scaling) is the noise source")
        print(f"  ✓ Actual Jitter: {h_data['jitter_mean']:.2f} px")
        print(f"  ✓ Actual SNR: {h_data['snr']:.2f}")
    elif TEST_STAGE == 3:
        print("STAGE 3 (Physiological Benchmark - Living Subject):")
        print("  ✓ Expected: Jitter ~23.73 px, SNR ~7.29 (>5.0 ✓ Rose Criterion)")
        print("  ✓ Proves: 1D EMA filter + deadzone recovers usable signal")
        print(f"  ✓ Actual Jitter: {h_data['jitter_mean']:.2f} px")
        print(f"  ✓ Actual SNR: {h_data['snr']:.2f}")

    print("=" * 70 + "\n")

    # Save analysis
    analysis_filename = f"snr_analysis_stage{TEST_STAGE}_{timestamp}.json"
    analysis_output = {
        "stage": TEST_STAGE,
        "input_mode": INPUT_MODE,
        "timestamp": timestamp,
        "screen_resolution": {"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
        "analysis": analysis,
        "safe_zones": calculate_safe_zones(sigma_x, NUM_SAFE_ZONES),
        "fixation_data": fixation_log
    }
    with open(analysis_filename, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"► Analysis saved to: {analysis_filename}\n")
else:
    print("\nNo data collected.")

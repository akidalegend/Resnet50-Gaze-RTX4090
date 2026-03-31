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
import warnings
from datetime import datetime
import time

import glob


def get_latest_file(pattern, directory="."):
    """
    Find the most recent file matching a pattern.

    Args:
        pattern: Glob pattern (e.g., "calibration*.json", "gaze_log_*.json")
        directory: Directory to search in

    Returns:
        Path to the most recent file, or None if no matches
    """
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)

    if not files:
        return None

    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# =============================================================================
# RESOLUTION CONSTANTS - 4K Display Configuration
# =============================================================================
SCREEN_WIDTH = 3840   # 4K horizontal resolution
SCREEN_HEIGHT = 2160  # 4K vertical resolution
WEBCAM_WIDTH = 1920   # 1080p webcam
WEBCAM_HEIGHT = 1080

# 1D Pivot Configuration - Hard Lock Y to vertical center
Y_LOCK_POSITION = SCREEN_HEIGHT // 2  # 1080 px (center of 4K display)

# Fixation Test Configuration
FIXATION_FRAMES = 300  # ~10 seconds at 30fps
FIXATION_TARGET_X = SCREEN_WIDTH // 2   # 1920 px
FIXATION_TARGET_Y = SCREEN_HEIGHT // 2  # 1080 px

# Rose Criterion SNR Threshold (1948)
ROSE_SNR_THRESHOLD = 5.0

# 1D Safe Zone Configuration
NUM_SAFE_ZONES = 3  # Left, Center, Right


# =============================================================================
# SNR & PRECISION ANALYSIS FUNCTIONS
# =============================================================================
def calculate_snr(signal_range, noise_std):
    """
    Calculate Signal-to-Noise Ratio using Rose Criterion (1948).

    Args:
        signal_range: Full screen dimension (3840 for X, 2160 for Y)
        noise_std: Standard deviation of logged coordinates during fixation

    Returns:
        SNR value (Signal / Noise)
    """
    if noise_std == 0:
        return float('inf')
    return signal_range / noise_std


def analyze_fixation_data(log_data):
    """
    Perform comprehensive SNR and precision analysis on fixation data.

    Returns dict with analysis results including stability flags.
    """
    if len(log_data) < 10:
        return {"error": "Insufficient data for analysis"}

    mapped_x = np.array([entry['mapped_x'] for entry in log_data])
    mapped_y = np.array([entry['mapped_y'] for entry in log_data])
    raw_yaw = np.array([entry['raw_yaw'] for entry in log_data])
    raw_pitch = np.array([entry['raw_pitch'] for entry in log_data])

    # Calculate standard deviations (noise)
    sigma_x = np.std(mapped_x)
    sigma_y = np.std(mapped_y)
    sigma_raw_yaw = np.std(raw_yaw)
    sigma_raw_pitch = np.std(raw_pitch)

    # Calculate SNR using Rose Criterion
    snr_x = calculate_snr(SCREEN_WIDTH, sigma_x)
    snr_y = calculate_snr(SCREEN_HEIGHT, sigma_y)

    # Stability flags based on Rose Criterion
    x_stable = bool(snr_x >= ROSE_SNR_THRESHOLD)
    y_stable = bool(snr_y >= ROSE_SNR_THRESHOLD)

    # Frame-to-frame jitter
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
    """
    Calculate 1D horizontal safe zones with 6-sigma guard bands.

    Uses measured horizontal jitter to ensure noise doesn't cause
    false zone-crossings.

    Args:
        sigma_x: Standard deviation of horizontal coordinates
        num_zones: Number of zones (default 3: Left, Center, Right)

    Returns:
        List of zone definitions with boundaries and guard bands
    """
    zone_width = SCREEN_WIDTH / num_zones
    guard_band = 6 * sigma_x  # 6-sigma boundary

    zones = []
    zone_names = ["LEFT", "CENTER", "RIGHT"] if num_zones == 3 else [f"ZONE_{i}" for i in range(num_zones)]

    for i in range(num_zones):
        zone_start = i * zone_width
        zone_end = (i + 1) * zone_width

        # Safe boundaries account for guard bands
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
    """
    Determine which safe zone the current X position falls into.
    Returns zone name and confidence (inside safe boundary or guard band).
    """
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

# 1. Load Calibration Data (finds latest calibration*.json file)
calib_file = get_latest_file("calibration*.json")

if not calib_file:
    # Fallback to exact name
    if os.path.exists("calibration.json"):
        calib_file = "calibration.json"
    else:
        print("ERROR: No calibration file found! Run calibration.py first.")
        exit()

with open(calib_file, "r") as f:
    calib = json.load(f)
    print(f"Calibration loaded from: {calib_file}")
    print(f"  sens_x: {calib['sens_x']:.4f}")
    print(f"  off_x: {calib['off_x']:.4f}")

# 2. Setup Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = create_model('resnet50', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("gaze_resnet50_epoch10.pth", weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Setup logging
log_data = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"gaze_log_{timestamp}.json"

# 4. Setup Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# 5. Setup Camera (1080p)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
window_name = "ADHD Gaze Tracker - 4K 1D Pivot Mode"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
ret, dummy_frame = cap.read()
if ret: cv2.imshow(window_name, dummy_frame)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

# 6. Smoothing Queues
# Initialize session-specific bias (replaces hardcoded BIAS_X)
session_bias_x = 0 
history_x = deque(maxlen=5) # Ensure history exists for smoothing
history_y = deque(maxlen=7)
bbox_history_x = deque(maxlen=5)
bbox_history_y = deque(maxlen=5)
bbox_history_w = deque(maxlen=5)
bbox_history_h = deque(maxlen=5)

# 7. Mode State Machine
MODE_LIVE = 0
MODE_FIXATION_TEST = 1
MODE_ANALYSIS = 2

current_mode = MODE_LIVE
fixation_log = []
fixation_frame_count = 0
safe_zones = None  # Will be calculated after fixation test

print("\n" + "=" * 60)
print("ADHD GAZE TRACKER - 4K 1D PIVOT MODE")
print("=" * 60)
print(f"Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT} (4K)")
print(f"Y-Axis Lock: {Y_LOCK_POSITION} px (center)")
print(f"Fixation Target: ({FIXATION_TARGET_X}, {FIXATION_TARGET_Y})")
print("-" * 60)
print("Controls:")
print("  [F] - Start Fixation Test (300 frames)")
print("  [Q] - Quit")
print("=" * 60 + "\n")


# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Scale factor from webcam to 4K display
    scale_x = SCREEN_WIDTH / w
    scale_y = SCREEN_HEIGHT / h

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    # Default values
    target_x = SCREEN_WIDTH // 2
    target_y = Y_LOCK_POSITION
    raw_yaw = 0.0
    raw_pitch = 0.0

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box

            # Smooth the Bounding Box
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
                    # Must match calibration.py scaling (* 1000.0)
                    raw_pitch = output[0][0].item() * 1000.0
                    raw_yaw = output[0][1].item() * 1000.0

                # --- 1D HORIZONTAL PIVOT WITH AUTO-ZEROING ---
                # 1. Map raw yaw to 4K screen coordinates
                raw_target_x = (raw_yaw * calib['sens_x']) + calib['off_x']
                
                # 2. Apply the Dynamic Session Bias (calculated when 'Z' is pressed)
                corrected_x = raw_target_x + session_bias_x
                
                # 3. Apply Temporal Smoothing (Moving Average)
                history_x.append(corrected_x)
                target_x = int(sum(history_x) / len(history_x))

                # 4. Final Lock and Constrain
                target_y = Y_LOCK_POSITION
                target_x = max(0, min(SCREEN_WIDTH, target_x))

                # Calculate mapped_y for logging (even though we don't use it)
                mapped_y_raw = int((raw_pitch * calib['sens_y']) + calib['off_y'])
                mapped_y_raw = max(0, min(SCREEN_HEIGHT, mapped_y_raw))

    # Convert 4K coordinates to webcam display coordinates for visualization
    display_x = int(target_x / scale_x)
    display_y = int(target_y / scale_y)

    # Mode-specific rendering and logging
    if current_mode == MODE_FIXATION_TEST:
        # Draw fixation target
        fixation_display_x = int(FIXATION_TARGET_X / scale_x)
        fixation_display_y = int(FIXATION_TARGET_Y / scale_y)
        cv2.circle(frame, (fixation_display_x, fixation_display_y), 15, (0, 0, 255), -1)
        cv2.circle(frame, (fixation_display_x, fixation_display_y), 5, (255, 255, 255), -1)

        # Draw gaze point
        cv2.circle(frame, (display_x, display_y), 10, (0, 255, 0), -1)

        # Log fixation data
        fixation_log.append({
            "frame": fixation_frame_count,
            "timestamp": time.time(),
            "raw_yaw": raw_yaw,
            "raw_pitch": raw_pitch,
            "mapped_x": target_x,
            "mapped_y": mapped_y_raw if 'mapped_y_raw' in dir() else Y_LOCK_POSITION
        })
        fixation_frame_count += 1

        # Progress indicator
        progress = fixation_frame_count / FIXATION_FRAMES
        bar_width = int(w * 0.6)
        bar_x = int(w * 0.2)
        cv2.rectangle(frame, (bar_x, 30), (bar_x + bar_width, 50), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, 30), (bar_x + int(bar_width * progress), 50), (0, 255, 0), -1)
        cv2.putText(frame, f"Fixation Test: {fixation_frame_count}/{FIXATION_FRAMES}",
                    (bar_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Check if fixation test complete
        if fixation_frame_count >= FIXATION_FRAMES:
            current_mode = MODE_ANALYSIS
            print("\nFixation test complete! Analyzing data...")

    elif current_mode == MODE_ANALYSIS:
        # Perform analysis
        analysis = analyze_fixation_data(fixation_log)

        # Calculate safe zones based on measured jitter
        sigma_x = analysis['horizontal']['std_sigma']
        safe_zones = calculate_safe_zones(sigma_x, NUM_SAFE_ZONES)

        # Print analysis report
        print("\n" + "=" * 60)
        print("SNR & PRECISION ANALYSIS REPORT (Rose Criterion 1948)")
        print("=" * 60)
        print(f"Frames Analyzed: {analysis['num_frames']}")
        print(f"Rose SNR Threshold: {ROSE_SNR_THRESHOLD}")
        print("-" * 60)

        print("\nHORIZONTAL AXIS (X):")
        h_data = analysis['horizontal']
        print(f"  Mean: {h_data['mean']:.1f} px")
        print(f"  Std Dev (sigma): {h_data['std_sigma']:.2f} px")
        print(f"  SNR (Signal/Noise): {h_data['snr']:.2f}")
        print(f"  Status: {h_data['status']}")
        print(f"  Range: [{h_data['range'][0]:.0f}, {h_data['range'][1]:.0f}] px")
        print(f"  Frame-to-Frame Jitter: {h_data['jitter_mean']:.2f} px")

        print("\nVERTICAL AXIS (Y) - Locked at center:")
        v_data = analysis['vertical']
        print(f"  Mean: {v_data['mean']:.1f} px")
        print(f"  Std Dev (sigma): {v_data['std_sigma']:.2f} px")
        print(f"  SNR (Signal/Noise): {v_data['snr']:.2f}")
        print(f"  Status: {v_data['status']}")
        print(f"  NOTE: Y-axis locked - vertical SNR shown for reference only")

        print("\n1D SAFE ZONES (6-sigma guard bands):")
        print(f"  Guard Band Width: {safe_zones[0]['guard_band_px']:.1f} px")
        for zone in safe_zones:
            print(f"  {zone['name']}: Raw [{zone['raw_boundary'][0]:.0f}-{zone['raw_boundary'][1]:.0f}] "
                  f"Safe [{zone['safe_boundary'][0]:.0f}-{zone['safe_boundary'][1]:.0f}]")

        print("=" * 60)

        # Save analysis to file
        analysis_filename = f"snr_analysis_{timestamp}.json"
        analysis_output = {
            "timestamp": timestamp,
            "screen_resolution": {"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
            "fixation_target": {"x": FIXATION_TARGET_X, "y": FIXATION_TARGET_Y},
            "analysis": analysis,
            "safe_zones": safe_zones,
            "fixation_data": fixation_log
        }
        with open(analysis_filename, "w") as f:
            json.dump(analysis_output, f, indent=2)
        print(f"\nAnalysis saved to: {analysis_filename}")

        # Return to live mode
        current_mode = MODE_LIVE
        fixation_log = []
        fixation_frame_count = 0

    else:  # MODE_LIVE
        # Draw 1D gaze point (green)
        cv2.circle(frame, (display_x, display_y), 20, (0, 255, 0), -1)

        # Draw safe zone indicators if available
        if safe_zones:
            zone_name, confidence = get_current_zone(target_x, safe_zones)
            color = (0, 255, 0) if confidence == "HIGH_CONFIDENCE" else (0, 255, 255)
            cv2.putText(frame, f"Zone: {zone_name} ({confidence})",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw zone boundaries on display
            for zone in safe_zones:
                boundary_x = int(zone['raw_boundary'][1] / scale_x)
                if boundary_x < w:
                    cv2.line(frame, (boundary_x, 0), (boundary_x, h), (100, 100, 100), 1)

        # Log data for standard tracking
        log_data.append({
            "frame": frame_count,
            "timestamp": time.time(),
            "raw_yaw": raw_yaw,
            "raw_pitch": raw_pitch,
            "target_x": target_x,
            "target_y": target_y,
            "raw_target_x": target_x,
            "raw_target_y": mapped_y_raw if 'mapped_y_raw' in dir() else Y_LOCK_POSITION
        })

    # Display info overlay
    cv2.putText(frame, f"4K Mode: {SCREEN_WIDTH}x{SCREEN_HEIGHT} | Y-Lock: {Y_LOCK_POSITION}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Gaze: ({target_x}, {target_y}) | Raw Yaw: {raw_yaw:.4f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "[F] Fixation Test | [Q] Quit",
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(window_name, frame)
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    
    # --- AUTO-ZEROING COMMAND ---
    # When 'Z' is pressed, calculate the difference between current gaze and center
    if key == ord('z'):
        # 1920 is the center of your 4K screen
        # We use the raw_target_x (unsmoothed) to get an instant anchor
        session_bias_x = 1920 - raw_target_x 
        print(f"System Zeroed! New Session Bias: {session_bias_x:.2f} px")
    
    if key == ord('q'):
        break
    elif key == ord('f') and current_mode == MODE_LIVE:
        current_mode = MODE_FIXATION_TEST
        fixation_log = []
        fixation_frame_count = 0
        print("\nStarting Fixation Test - Look at the RED target...")

cap.release()
cv2.destroyAllWindows()

# Save final log file
if log_data:
    final_log = {
        "metadata": {
            "screen_resolution": {"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
            "y_lock_position": Y_LOCK_POSITION,
            "calibration": calib,
            "total_frames": len(log_data),
            "timestamp": timestamp
        },
        "points": log_data
    }
    with open(log_filename, "w") as f:
        json.dump(final_log, f, indent=2)
    print(f"\nLogged {len(log_data)} frames to {log_filename}")
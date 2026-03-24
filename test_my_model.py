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

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# 1. Load Calibration Data
if not os.path.exists("calibration.json"):
    print("❌ ERROR: You must run calibrate.py first!")
    exit()

with open("calibration.json", "r") as f:
    calib = json.load(f)
    print("✅ Calibration data loaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('resnet50', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("gaze_resnet50_epoch10.pth", weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Setup logging
log_data = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"gaze_log_{timestamp}.json"

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
window_name = "ADHD Gaze Tracker - Live Test"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
ret, dummy_frame = cap.read()
if ret: cv2.imshow(window_name, dummy_frame)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

# Smoothing Queues
history_x = deque(maxlen=7)
history_y = deque(maxlen=7)
bbox_history_x = deque(maxlen=5)
bbox_history_y = deque(maxlen=5)
bbox_history_w = deque(maxlen=5)
bbox_history_h = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    

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
                    # REMOVE the * 1000.0 multiplier for now
                    pred_pitch = output[0][0].item() 
                    pred_yaw = output[0][1].item()
                    
                # 1. RAW DATA CHECK: Print these to your console
                print(f"RAW -> Pitch: {pred_pitch:.4f} | Yaw: {pred_yaw:.4f}")

                # --- 1D HORIZONTAL PIVOT ---
                # Use the raw yaw for horizontal tracking
                target_x = int((pred_yaw * calib['sens_x']) + calib['off_x'])

                # LOCK Y to the vertical center of your screen (e.g., 540 for a 1080p monitor)
                target_y = 540 

                # Constrain X to stay within screen bounds
                target_x = max(0, min(w, target_x))

                # Draw the 1D Gaze Dot
                cv2.circle(frame, (target_x, target_y), 20, (0, 255, 0), -1) # Green for 1D mode

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# Save log file
if log_data:
    with open(log_filename, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n✅ Logged {len(log_data)} frames to {log_filename}")
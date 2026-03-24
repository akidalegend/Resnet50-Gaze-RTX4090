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
                    pred_pitch = output[0][0].item() * 1000.0
                    pred_yaw = output[0][1].item() * 1000.0
                    print(f"Looking at Camera -> Yaw: {pred_yaw:.1f} | Pitch: {pred_pitch:.1f}")

                # --- 1. APPLY JSON CALIBRATION ---
                raw_target_x = (pred_yaw * calib['sens_x']) + calib['off_x']
                raw_target_y = (pred_pitch * calib['sens_y']) + calib['off_y']
                
                # --- 2. DYNAMIC ANCHOR TRACKING (Moving Head Fix) ---
                # How far did you move from your calibrated sitting position?
                head_shift_x = face_center_x - calib['anchor_x']
                head_shift_y = face_center_y - calib['anchor_y']
                
                # Compensate for the physical head movement!
                target_x = int(raw_target_x - head_shift_x)
                target_y = int(raw_target_y - head_shift_y)
                
                # Apply Smoothing
                history_x.append(target_x)
                history_y.append(target_y)
                smooth_x = int(sum(history_x) / len(history_x))
                smooth_y = int(sum(history_y) / len(history_y))
                
                smooth_x = max(0, min(w, smooth_x))
                smooth_y = max(0, min(h, smooth_y))
                
                cv2.circle(frame, (smooth_x, smooth_y), 20, (0, 0, 255), -1) 
                cv2.circle(frame, (smooth_x, smooth_y), 5, (255, 255, 255), -1)
                cv2.line(frame, (face_center_x, face_center_y), (smooth_x, smooth_y), (0, 255, 0), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
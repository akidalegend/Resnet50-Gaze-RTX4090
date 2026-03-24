import torch
import cv2
import json
import mediapipe as mp
from timm import create_model
from torchvision import transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# 1. Setup Model (Same as your live tracker)
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
window_name = "ADHD Gaze - Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
ret, dummy_frame = cap.read()
if ret: cv2.imshow(window_name, dummy_frame)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

# Calibration State Machine
state = 0
calibration_data = {}
pred_tl_x, pred_tl_y = 0, 0
pred_br_x, pred_br_y = 0, 0

print("🎯 Starting Calibration...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    display_frame = frame.copy()
    
    # Process Face and AI Prediction
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    pred_yaw, pred_pitch = 0, 0
    face_center_x, face_center_y = w//2, h//2
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            xmin = int(bboxC.xmin * w)
            ymin = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            
            pad_x, pad_y = int(width * 0.2), int(height * 0.2)
            x1, y1 = max(0, xmin - pad_x), max(0, ymin - pad_y)
            x2, y2 = min(w, xmin + width + pad_x), min(h, ymin + height + pad_y)
            
            face_center_x = x1 + (x2 - x1) // 2
            face_center_y = y1 + (y2 - y1) // 2
            
            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size != 0:
                input_tensor = transform(Image.fromarray(face_crop)).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_pitch = output[0][0].item() * 1000.0
                    pred_yaw = output[0][1].item() * 1000.0

    # Draw UI based on State
    # We use a 100 pixel margin from the edges
    margin_x = w // 4  # 480 pixels in from the left/right
    margin_y = h // 4  # 270 pixels in from the top/bottom
    
    if state == 0:
        cv2.putText(display_frame, "Look at the RED DOT and press SPACE", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.circle(display_frame, (margin_x, margin_y), 20, (0, 0, 255), -1) # Top Left
    elif state == 1:
        cv2.circle(display_frame, (w - margin_x, h - margin_y), 20, (0, 0, 255), -1) # Bottom Right
    elif state == 2:
        cv2.circle(display_frame, (w//2, h//2), 20, (0, 0, 255), -1) # Center
    elif state == 3:
        cv2.putText(display_frame, "Calibration Complete! Saving...", (w//3, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        if state == 0:
            pred_tl_x, pred_tl_y = pred_yaw, pred_pitch
            print(f"Top-Left Captured: {pred_tl_x:.1f}, {pred_tl_y:.1f}")
            state = 1
        elif state == 1:
            pred_br_x, pred_br_y = pred_yaw, pred_pitch
            print(f"Bottom-Right Captured: {pred_br_x:.1f}, {pred_br_y:.1f}")
            state = 2
        elif state == 2:
            calibration_data['anchor_x'] = face_center_x
            calibration_data['anchor_y'] = face_center_y
            
            diff_x = pred_br_x - pred_tl_x if pred_br_x != pred_tl_x else 1
            diff_y = pred_br_y - pred_tl_y if pred_br_y != pred_tl_y else 1
            
            # Update the algebra to use the new margins
            screen_diff_x = (w - margin_x) - margin_x
            screen_diff_y = (h - margin_y) - margin_y
            
            calibration_data['sens_x'] = screen_diff_x / diff_x
            calibration_data['sens_y'] = screen_diff_y / diff_y
            calibration_data['off_x'] = margin_x - (pred_tl_x * calibration_data['sens_x'])
            calibration_data['off_y'] = margin_y - (pred_tl_y * calibration_data['sens_y'])
            
            with open("calibration.json", "w") as f:
                json.dump(calibration_data, f)
            print("✅ calibration.json saved!")
            state = 3
            
    elif key == ord('q') or state == 3:
        if state == 3: cv2.waitKey(1500) # Pause to let user read the complete message
        break

cap.release()
cv2.destroyAllWindows()
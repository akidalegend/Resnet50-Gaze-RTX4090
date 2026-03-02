import torch
import cv2
import numpy as np
from timm import create_model
from torchvision import transforms
from PIL import Image

# 1. Hardware and Model Setup
device = torch.device("cuda")
print(f"Loading weights onto: {torch.cuda.get_device_name(0)}")

# Initialize the exact same architecture used in training
model = create_model('resnet50', pretrained=False, num_classes=2)

# Load your Epoch 1 weights!
weights_path = "gaze_resnet50_epoch1.pth"
model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval() # Set to evaluation mode (turns off learning)
print("✅ Weights loaded successfully!")

# 2. Image Pre-processing (Must match training exactly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Webcam active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror the frame so it acts like a normal mirror
    frame = cv2.flip(frame, 1)
    
    # Pre-process frame for the AI
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Add batch dimension: [1, 3, 224, 224]
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # 4. Get the AI's Prediction
    with torch.no_grad():
        output = model(input_tensor)
        
        # --- THE DE-NORMALIZATION STEP ---
        # Multiply by 1000 to get the original scale back!
        pred_pitch = output[0][0].item() * 1000.0
        pred_yaw = output[0][1].item() * 1000.0

    # 5. Visualizing the Gaze
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    
    # Map the predictions to a visual arrow
    # (Scaling by a small factor so the arrow fits on screen)
    dx = int(pred_yaw * 0.5)
    dy = int(pred_pitch * 0.5)
    
    # Draw the Gaze Direction Arrow
    cv2.arrowedLine(frame, (center_x, center_y), (center_x + dx, center_y + dy), (0, 255, 0), 4, tipLength=0.2)
    
    # Print the raw numbers on the screen
    cv2.putText(frame, f"Pitch (Y): {pred_pitch:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Yaw (X): {pred_yaw:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("ADHD Gaze Tracker - Live Test", frame)
    
    # Press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
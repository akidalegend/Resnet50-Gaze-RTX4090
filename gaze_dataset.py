import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MPIIGazeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.gaze_frame = pd.read_csv(csv_file)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.gaze_frame)

    def __getitem__(self, idx):
        img_path = str(self.gaze_frame.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        
        # --- SCALE TARGETS TO [0, 1] ---
        # 606 / 1000 = 0.606
        # 981 / 1000 = 0.981
        pitch = float(self.gaze_frame.iloc[idx, 1]) / 1000.0 
        yaw = float(self.gaze_frame.iloc[idx, 2]) / 1000.0
        
        labels = torch.tensor([pitch, yaw], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels
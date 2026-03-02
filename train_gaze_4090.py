import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm import create_model
from gaze_dataset import MPIIGazeDataset 
import time
import os
import matplotlib.pyplot as plt

def train():
    # 1. Hardware Setup
    device = torch.device("cuda")
    print(f"🚀 Training on: {torch.cuda.get_device_name(0)}")

    # 2. Hyperparameters
    # Optimized for RTX 4090 24GB VRAM
    BATCH_SIZE = 256  
    EPOCHS = 10       
    LEARNING_RATE = 0.0001
    
    # Path to your clean CSV and project root
    ROOT_DIR = r"C:\Users\23828140\Documents\GitHub\Resnet50-Gaze-RTX4090"
    CSV_PATH = os.path.join(ROOT_DIR, "dataset", "clean_labels.csv")

    # 3. Load Data
    print("Loading dataset...")
    train_loader = DataLoader(
        MPIIGazeDataset(csv_file=CSV_PATH), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,    # Keep at 4 for high-speed image loading
        pin_memory=True
    )

    # 4. Initialize Model (ResNet-50)
    print("Initializing ResNet-50 architecture...")
    model = create_model('resnet50', pretrained=True, num_classes=2)
    model.to(device)

    # 5. Loss and Optimizer
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Tracking for Dissertation Graphs
    epoch_losses = []

    # 7. Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, labels) in enumerate(train_loader):
            # Move data to 4090
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 100 steps
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        epoch_time = time.time() - start_time
        
        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # --- SAVE RESULTS ---
        # 1. Save Weights
        torch.save(model.state_dict(), f"gaze_resnet50_epoch{epoch+1}.pth")
        
        # 2. Update Loss Graph
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o', color='b', label='Training Loss')
        plt.title('ADHD Gaze Model: Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('learning_curve.png')
        plt.close()

    print("Training Finished! Final model saved as: gaze_resnet50_epoch10.pth")

# Critical Windows Multiprocessing Guard
if __name__ == '__main__':
    train()
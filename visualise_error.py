import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def main():
    # Get log file from command line or use default
    if len(sys.argv) > 1:
        FILE_PATH = sys.argv[1]
    else:
        # Find the most recent gaze_log
        logs = [f for f in os.listdir('.') if f.startswith('gaze_log_') and f.endswith('.json')]
        if not logs:
            print("Error: No gaze_log files found")
            return
        FILE_PATH = sorted(logs)[-1]
    
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both array and dict formats
    if isinstance(data, list):
        log_entries = data
    else:
        log_entries = data.get("points", [])

    if not log_entries:
        print("Error: No data found in log file")
        return

    # Extract prediction data from log entries
    pred_x_vals = []
    pred_y_vals = []
    raw_pred_x = []
    raw_pred_y = []
    
    for entry in log_entries:
        # Use the calculated target coordinates (after calibration)
        if "target_x" in entry and "target_y" in entry:
            pred_x_vals.append(entry["target_x"])
            pred_y_vals.append(entry["target_y"])
        
        # Also track raw values
        if "raw_target_x" in entry and "raw_target_y" in entry:
            raw_pred_x.append(entry["raw_target_x"])
            raw_pred_y.append(entry["raw_target_y"])

    print(f"📊 Analyzing {len(log_entries)} frames from {FILE_PATH}\n")

    # Calculate statistics
    pred_x_vals = np.array(pred_x_vals)
    pred_y_vals = np.array(pred_y_vals)
    
    # Calculate stability metrics
    jitter_x = np.mean(np.abs(np.diff(pred_x_vals)))
    jitter_y = np.mean(np.abs(np.diff(pred_y_vals)))
    variance_x = np.var(pred_x_vals)
    variance_y = np.var(pred_y_vals)
    
    print("=" * 60)
    print("PREDICTION STATISTICS")
    print("=" * 60)
    print(f"X Range: {np.min(pred_x_vals):.0f} - {np.max(pred_x_vals):.0f} px")
    print(f"Y Range: {np.min(pred_y_vals):.0f} - {np.max(pred_y_vals):.0f} px")
    print(f"X Mean: {np.mean(pred_x_vals):.0f} px | Std: {np.std(pred_x_vals):.1f} px")
    print(f"Y Mean: {np.mean(pred_y_vals):.0f} px | Std: {np.std(pred_y_vals):.1f} px")
    print(f"X Jitter (frame-to-frame): {jitter_x:.1f} px")
    print(f"Y Jitter (frame-to-frame): {jitter_y:.1f} px")
    print("=" * 60 + "\n")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Spatial Distribution (Heatmap-style scatter)
    ax1 = axes[0, 0]
    hist, xedges, yedges = np.histogram2d(pred_x_vals, pred_y_vals, bins=30)
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    im = ax1.imshow(hist.T, extent=extent, origin='upper', cmap='hot', aspect='auto')
    ax1.set_xlabel("Screen X (pixels)")
    ax1.set_ylabel("Screen Y (pixels)")
    ax1.set_title("Gaze Distribution Heatmap\n(Red = most predictions)")
    plt.colorbar(im, ax=ax1, label='Frequency')
    
    # Plot 2: Timeline - X coordinate
    ax2 = axes[0, 1]
    ax2.plot(pred_x_vals, label='Target X', linewidth=1, alpha=0.7)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("X Coordinate (pixels)")
    ax2.set_title("X Coordinate Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Timeline - Y coordinate
    ax3 = axes[1, 0]
    ax3.plot(pred_y_vals, label='Target Y', color='orange', linewidth=1, alpha=0.7)
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Y Coordinate (pixels)")
    ax3.set_title("Y Coordinate Over Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Jitter visualization (frame-to-frame movement)
    ax4 = axes[1, 1]
    dx = np.diff(pred_x_vals)
    dy = np.diff(pred_y_vals)
    jitter_magnitude = np.sqrt(dx**2 + dy**2)
    ax4.plot(jitter_magnitude, label='Frame-to-Frame Distance', color='red', linewidth=1, alpha=0.7)
    ax4.axhline(y=np.mean(jitter_magnitude), color='r', linestyle='--', label=f'Mean: {np.mean(jitter_magnitude):.1f}px')
    ax4.set_xlabel("Frame")
    ax4.set_ylabel("Distance (pixels)")
    ax4.set_title("Jitter (Movement Between Frames)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
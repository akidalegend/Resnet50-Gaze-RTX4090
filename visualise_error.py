import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path to the calibration file
# You can change this to test_2 or any other session file
FILE_PATH = "sessions/calibration/akrs_test_1_jan_16_calibration.json"

def apply_poly2(model, h, v):
    """Replicates the calibration mapping to convert raw gaze to screen pixels."""
    if model.get("type") == "poly2":
        feats = np.array([1.0, h, v, h * v, h ** 2, v ** 2], dtype=float)
        x = np.dot(np.array(model["x_coef"], dtype=float), feats)
        y = np.dot(np.array(model["y_coef"], dtype=float), feats)
        return x, y
    else:
        # Linear fallback
        x = model["x_slope"] * h + model["x_intercept"]
        y = model["y_slope"] * v + model["y_intercept"]
        return x, y

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = data.get("points", [])
    model = data.get("model", {})
    screen_w = data.get("screen_width", 1920)
    screen_h = data.get("screen_height", 1080)

    if not model or not points:
        print("Error: Missing model or points data in JSON.")
        return

    # Prepare data arrays
    targets_x, targets_y = [], []
    preds_x, preds_y = [], []
    errors = []

    for p in points:
        if p.get("gaze_h") is not None and p.get("gaze_v") is not None:
            tx, ty = p["target_x"], p["target_y"]
            gh, gv = p["gaze_h"], p["gaze_v"]
            
            px, py = apply_poly2(model, gh, gv)
            
            targets_x.append(tx)
            targets_y.append(ty)
            preds_x.append(px)
            preds_y.append(py)
            
            # Calculate Euclidean distance (error in pixels)
            dist = np.sqrt((px - tx)**2 + (py - ty)**2)
            errors.append(dist)

    # Calculate global metrics
    rmse = np.sqrt(np.mean(np.square(errors)))

    # Generate the Visualization
    plt.figure(figsize=(10, 6))
    plt.title(f"Spatial Gaze Error Map\nGlobal RMSE: {rmse:.1f} px", fontsize=14)
    
    # Invert Y axis to match screen coordinates (0,0 is top-left)
    plt.gca().invert_yaxis()
    plt.xlim(0, screen_w)
    plt.ylim(screen_h, 0)
    
    # Plot target points (Ideal)
    plt.scatter(targets_x, targets_y, c='blue', marker='+', s=200, label='True Targets', zorder=3)
    
    # Plot predicted points (Actual)
    plt.scatter(preds_x, preds_y, c='red', marker='o', s=50, label='Model Estimation', zorder=3)
    
    # Draw error lines connecting target to prediction
    for tx, ty, px, py, err in zip(targets_x, targets_y, preds_x, preds_y, errors):
        plt.plot([tx, px], [ty, py], 'k--', alpha=0.5, zorder=2)
        
        # Annotate each line with its specific error value
        mid_x = (tx + px) / 2
        mid_y = (ty + py) / 2
        plt.text(mid_x + 15, mid_y, f"{err:.1f}px", color='black', fontsize=9)

    # Grid and formatting
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel("Screen Width (pixels)")
    plt.ylabel("Screen Height (pixels)")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
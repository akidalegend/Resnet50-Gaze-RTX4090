"""
Compare two gaze prediction logs to evaluate raw vs compensated values.
Usage: python compare_logs.py <log_file_1> <log_file_2>
"""

import json
import numpy as np
import sys
from pathlib import Path

def calculate_metrics(log_data):
    """Calculate stability and accuracy metrics from log data."""
    if not log_data:
        return None
    
    smooth_x = np.array([entry["smooth_x"] for entry in log_data])
    smooth_y = np.array([entry["smooth_y"] for entry in log_data])
    target_x = np.array([entry["target_x"] for entry in log_data])
    target_y = np.array([entry["target_y"] for entry in log_data])
    
    # Jitter: variance of frame-to-frame differences
    dx = np.diff(smooth_x)
    dy = np.diff(smooth_y)
    jitter = np.sqrt(np.mean(dx**2 + dy**2))
    
    # Variance: spread of points
    variance_x = np.var(smooth_x)
    variance_y = np.var(smooth_y)
    total_variance = variance_x + variance_y
    
    # Smoothness: average absolute change per frame
    smoothness = np.mean(np.abs(dx)) + np.mean(np.abs(dy))
    
    # Range coverage
    range_x = np.max(smooth_x) - np.min(smooth_x)
    range_y = np.max(smooth_y) - np.min(smooth_y)
    
    # Drift: distance from start to end position
    drift = np.sqrt((smooth_x[-1] - smooth_x[0])**2 + (smooth_y[-1] - smooth_y[0])**2)
    
    return {
        "jitter": jitter,
        "variance": total_variance,
        "smoothness": smoothness,
        "range_x": range_x,
        "range_y": range_y,
        "drift": drift,
        "mean_x": np.mean(smooth_x),
        "mean_y": np.mean(smooth_y),
        "std_x": np.std(smooth_x),
        "std_y": np.std(smooth_y),
        "frame_count": len(log_data)
    }

def load_log(filename):
    """Load a gaze log file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in: {filename}")
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_logs.py <log_file_1> <log_file_2>")
        print("\nExample:")
        print("  python compare_logs.py gaze_log_raw.json gaze_log_compensated.json")
        print("\nMetrics explained:")
        print("  - Jitter: Frame-to-frame movement (lower = more stable)")
        print("  - Variance: Spread of predictions (lower = more consistent)")
        print("  - Smoothness: Total movement per frame (lower = smoother)")
        print("  - Drift: Distance from start to end (lower = more stable)")
        return
    
    log_file_1 = sys.argv[1]
    log_file_2 = sys.argv[2]
    
    print(f"\n📊 Comparing gaze logs...")
    print(f"   File 1: {log_file_1}")
    print(f"   File 2: {log_file_2}\n")
    
    # Load logs
    log1 = load_log(log_file_1)
    log2 = load_log(log_file_2)
    
    if not log1 or not log2:
        return
    
    # Calculate metrics
    metrics1 = calculate_metrics(log1)
    metrics2 = calculate_metrics(log2)
    
    if not metrics1 or not metrics2:
        print("❌ Could not calculate metrics")
        return
    
    # Display comparison
    print("=" * 70)
    print(f"{'Metric':<20} {'File 1':<15} {'File 2':<15} {'Winner':<15}")
    print("=" * 70)
    
    # Lower is better for jitter, variance, smoothness, drift
    metrics_to_compare = [
        ("Jitter (px/frame)", "jitter", True),
        ("Variance (px²)", "variance", True),
        ("Smoothness (px/frame)", "smoothness", True),
        ("Drift (px)", "drift", True),
        ("X Range (px)", "range_x", False),
        ("Y Range (px)", "range_y", False),
        ("X Mean (px)", "mean_x", False),
        ("Y Mean (px)", "mean_y", False),
        ("X Std Dev", "std_x", True),
        ("Y Std Dev", "std_y", True),
    ]
    
    wins = [0, 0]
    
    for display_name, metric_key, lower_is_better in metrics_to_compare:
        val1 = metrics1[metric_key]
        val2 = metrics2[metric_key]
        
        if lower_is_better:
            winner = "File 1 ✓" if val1 < val2 else "File 2 ✓" if val2 < val1 else "Tie"
            if val1 < val2:
                wins[0] += 1
            elif val2 < val1:
                wins[1] += 1
        else:
            winner = ""
        
        print(f"{display_name:<20} {val1:<15.2f} {val2:<15.2f} {winner:<15}")
    
    print("=" * 70)
    print(f"\nFrame counts: File 1 = {metrics1['frame_count']}, File 2 = {metrics2['frame_count']}")
    print(f"Overall: File 1 wins {wins[0]} metrics, File 2 wins {wins[1]} metrics")
    
    if wins[0] > wins[1]:
        print("\n🎯 File 1 appears to be MORE STABLE/ACCURATE")
    elif wins[1] > wins[0]:
        print("\n🎯 File 2 appears to be MORE STABLE/ACCURATE")
    else:
        print("\n⚖️  Metrics are roughly equivalent")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

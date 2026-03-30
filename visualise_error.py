import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# =============================================================================
# 4K RESOLUTION CONSTANTS
# =============================================================================
SCREEN_WIDTH = 3840   # 4K horizontal resolution
SCREEN_HEIGHT = 2160  # 4K vertical resolution

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


def calculate_safe_zones(sigma_x, screen_width=SCREEN_WIDTH, num_zones=NUM_SAFE_ZONES):
    """
    Calculate 1D horizontal safe zones with 6-sigma guard bands.

    Uses measured horizontal jitter to ensure noise doesn't cause
    false zone-crossings.

    Args:
        sigma_x: Standard deviation of horizontal coordinates
        screen_width: Screen width in pixels
        num_zones: Number of zones (default 3: Left, Center, Right)

    Returns:
        List of zone definitions with boundaries and guard bands
    """
    zone_width = screen_width / num_zones
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
            "safe_boundary": [max(0, safe_start), min(screen_width, safe_end)],
            "guard_band_px": guard_band,
            "usable_width": max(0, safe_end - safe_start)
        })

    return zones


def analyze_snr_and_precision(pred_x_vals, pred_y_vals, raw_yaw=None, raw_pitch=None,
                               screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT):
    """
    Perform comprehensive SNR and precision analysis.

    Args:
        pred_x_vals: Array of mapped X coordinates
        pred_y_vals: Array of mapped Y coordinates
        raw_yaw: Optional array of raw yaw predictions
        raw_pitch: Optional array of raw pitch predictions
        screen_width: Display width (default 4K)
        screen_height: Display height (default 4K)

    Returns:
        Dictionary containing full analysis results
    """
    # Calculate standard deviations (noise)
    sigma_x = np.std(pred_x_vals)
    sigma_y = np.std(pred_y_vals)

    # Calculate SNR using Rose Criterion (Signal = full screen dimension)
    snr_x = calculate_snr(screen_width, sigma_x)
    snr_y = calculate_snr(screen_height, sigma_y)

    # Stability assessment
    x_stable = snr_x >= ROSE_SNR_THRESHOLD
    y_stable = snr_y >= ROSE_SNR_THRESHOLD

    # Frame-to-frame jitter
    jitter_x = np.mean(np.abs(np.diff(pred_x_vals))) if len(pred_x_vals) > 1 else 0
    jitter_y = np.mean(np.abs(np.diff(pred_y_vals))) if len(pred_y_vals) > 1 else 0

    # Calculate safe zones based on horizontal sigma
    safe_zones = calculate_safe_zones(sigma_x, screen_width, NUM_SAFE_ZONES)

    # Total usable screen after guard bands
    total_usable = sum(z['usable_width'] for z in safe_zones)
    usable_percentage = (total_usable / screen_width) * 100

    analysis = {
        "num_samples": len(pred_x_vals),
        "screen_resolution": {"width": screen_width, "height": screen_height},
        "rose_threshold": ROSE_SNR_THRESHOLD,
        "horizontal": {
            "mean": float(np.mean(pred_x_vals)),
            "std_sigma": float(sigma_x),
            "variance": float(np.var(pred_x_vals)),
            "snr": float(snr_x),
            "snr_ratio": f"{snr_x:.1f}:1",
            "stable": x_stable,
            "status": "STABLE" if x_stable else "MATHEMATICALLY UNSTABLE",
            "jitter_mean": float(jitter_x),
            "range": [float(np.min(pred_x_vals)), float(np.max(pred_x_vals))]
        },
        "vertical": {
            "mean": float(np.mean(pred_y_vals)),
            "std_sigma": float(sigma_y),
            "variance": float(np.var(pred_y_vals)),
            "snr": float(snr_y),
            "snr_ratio": f"{snr_y:.1f}:1",
            "stable": y_stable,
            "status": "STABLE" if y_stable else "MATHEMATICALLY UNSTABLE",
            "jitter_mean": float(jitter_y),
            "range": [float(np.min(pred_y_vals)), float(np.max(pred_y_vals))]
        },
        "safe_zones": safe_zones,
        "usable_screen_percentage": usable_percentage,
        "guard_band_6sigma": float(6 * sigma_x),
        "clinical_recommendation": get_clinical_recommendation(snr_x, snr_y)
    }

    # Add raw model output stats if provided
    if raw_yaw is not None and raw_pitch is not None:
        analysis["raw_model_output"] = {
            "yaw_std": float(np.std(raw_yaw)),
            "pitch_std": float(np.std(raw_pitch)),
            "yaw_range": [float(np.min(raw_yaw)), float(np.max(raw_yaw))],
            "pitch_range": [float(np.min(raw_pitch)), float(np.max(raw_pitch))]
        }

    return analysis


def get_clinical_recommendation(snr_x, snr_y):
    """
    Generate clinical recommendation based on SNR values.
    """
    if snr_x >= ROSE_SNR_THRESHOLD and snr_y >= ROSE_SNR_THRESHOLD:
        return "FULL 2D TRACKING: Both axes meet Rose Criterion. Suitable for clinical use."
    elif snr_x >= ROSE_SNR_THRESHOLD:
        return "1D HORIZONTAL ONLY: Vertical axis unstable. Use Y-axis hard lock for clinical assessment."
    elif snr_y >= ROSE_SNR_THRESHOLD:
        return "1D VERTICAL ONLY: Horizontal axis unstable (unusual). Investigate calibration."
    else:
        return "NOT SUITABLE: Neither axis meets minimum SNR. Recalibrate or check hardware."


def print_snr_report(analysis):
    """
    Print formatted SNR analysis report to console.
    """
    print("\n" + "=" * 70)
    print("SNR & PRECISION ANALYSIS REPORT (Rose Criterion 1948)")
    print("=" * 70)
    print(f"Samples Analyzed: {analysis['num_samples']}")
    print(f"Screen Resolution: {analysis['screen_resolution']['width']}x{analysis['screen_resolution']['height']}")
    print(f"Rose SNR Threshold: >= {analysis['rose_threshold']} (for diagnostic reliability)")
    print("-" * 70)

    h = analysis['horizontal']
    print("\nHORIZONTAL AXIS (X):")
    print(f"  Signal (Screen Width): {analysis['screen_resolution']['width']} px")
    print(f"  Noise (Std Dev sigma): {h['std_sigma']:.2f} px")
    print(f"  SNR (Signal/Noise):    {h['snr']:.2f} ({h['snr_ratio']})")
    print(f"  Status:                {h['status']}")
    print(f"  Mean Position:         {h['mean']:.1f} px")
    print(f"  Range:                 [{h['range'][0]:.0f}, {h['range'][1]:.0f}] px")
    print(f"  Frame-to-Frame Jitter: {h['jitter_mean']:.2f} px")

    v = analysis['vertical']
    print("\nVERTICAL AXIS (Y):")
    print(f"  Signal (Screen Height): {analysis['screen_resolution']['height']} px")
    print(f"  Noise (Std Dev sigma):  {v['std_sigma']:.2f} px")
    print(f"  SNR (Signal/Noise):     {v['snr']:.2f} ({v['snr_ratio']})")
    print(f"  Status:                 {v['status']}")
    print(f"  Mean Position:          {v['mean']:.1f} px")
    print(f"  Range:                  [{v['range'][0]:.0f}, {v['range'][1]:.0f}] px")
    print(f"  Frame-to-Frame Jitter:  {v['jitter_mean']:.2f} px")

    print("\n" + "-" * 70)
    print("1D SAFE ZONES (6-sigma guard bands):")
    print(f"  Guard Band Width: {analysis['guard_band_6sigma']:.1f} px (6 * {h['std_sigma']:.2f})")
    print(f"  Usable Screen: {analysis['usable_screen_percentage']:.1f}%")
    print()
    for zone in analysis['safe_zones']:
        print(f"  {zone['name']:8} | Raw: [{zone['raw_boundary'][0]:7.0f} - {zone['raw_boundary'][1]:7.0f}] | "
              f"Safe: [{zone['safe_boundary'][0]:7.0f} - {zone['safe_boundary'][1]:7.0f}] | "
              f"Usable: {zone['usable_width']:.0f} px")

    print("\n" + "-" * 70)
    print("CLINICAL RECOMMENDATION:")
    print(f"  {analysis['clinical_recommendation']}")
    print("=" * 70 + "\n")


def create_snr_visualization(pred_x_vals, pred_y_vals, analysis, save_path=None):
    """
    Create comprehensive visualization including SNR analysis.
    """
    fig = plt.figure(figsize=(16, 12))

    # Define grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ----- Plot 1: Spatial Distribution with Safe Zones -----
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.scatter(pred_x_vals, pred_y_vals, alpha=0.3, s=10, c='blue', label='Gaze Points')
    ax1.axhline(y=np.mean(pred_y_vals), color='red', linestyle='--', alpha=0.5, label=f'Y Mean: {np.mean(pred_y_vals):.0f}')
    ax1.axvline(x=np.mean(pred_x_vals), color='green', linestyle='--', alpha=0.5, label=f'X Mean: {np.mean(pred_x_vals):.0f}')

    # Draw safe zone boundaries
    colors = ['#ffcccc', '#ccffcc', '#ccccff']
    for i, zone in enumerate(analysis['safe_zones']):
        ax1.axvspan(zone['raw_boundary'][0], zone['raw_boundary'][1],
                    alpha=0.15, color=colors[i % len(colors)], label=f"{zone['name']} Zone")
        # Draw guard bands
        if i > 0:
            ax1.axvline(x=zone['safe_boundary'][0], color='orange', linestyle=':', alpha=0.7)
        if i < len(analysis['safe_zones']) - 1:
            ax1.axvline(x=zone['safe_boundary'][1], color='orange', linestyle=':', alpha=0.7)

    ax1.set_xlabel("Screen X (pixels)")
    ax1.set_ylabel("Screen Y (pixels)")
    ax1.set_title("Gaze Distribution with Safe Zones\n(Dotted orange = 6σ guard bands)")
    ax1.set_xlim(0, SCREEN_WIDTH)
    ax1.set_ylim(SCREEN_HEIGHT, 0)  # Inverted for screen coordinates
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ----- Plot 2: SNR Bar Chart -----
    ax2 = fig.add_subplot(gs[0, 2])
    snr_values = [analysis['horizontal']['snr'], analysis['vertical']['snr']]
    bar_colors = ['green' if s >= ROSE_SNR_THRESHOLD else 'red' for s in snr_values]
    bars = ax2.bar(['Horizontal (X)', 'Vertical (Y)'], snr_values, color=bar_colors, edgecolor='black')
    ax2.axhline(y=ROSE_SNR_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Rose Threshold ({ROSE_SNR_THRESHOLD})')
    ax2.set_ylabel("SNR (Signal / Noise)")
    ax2.set_title("SNR by Axis\n(Green = Stable, Red = Unstable)")
    ax2.legend()
    for bar, val in zip(bars, snr_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # ----- Plot 3: X Coordinate Timeline -----
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(pred_x_vals, label='Mapped X', linewidth=0.8, alpha=0.7, color='blue')
    ax3.axhline(y=np.mean(pred_x_vals), color='green', linestyle='--',
                label=f'Mean: {np.mean(pred_x_vals):.0f}px')
    # Draw +/- 1 sigma bands
    ax3.fill_between(range(len(pred_x_vals)),
                     np.mean(pred_x_vals) - np.std(pred_x_vals),
                     np.mean(pred_x_vals) + np.std(pred_x_vals),
                     alpha=0.2, color='green', label=f'±1σ ({np.std(pred_x_vals):.1f}px)')
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("X Coordinate (pixels)")
    ax3.set_title(f"Horizontal Gaze Timeline | SNR: {analysis['horizontal']['snr']:.1f}:1")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # ----- Plot 4: Histogram of X values -----
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(pred_x_vals, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(pred_x_vals), color='red', linestyle='--',
                label=f'Mean: {np.mean(pred_x_vals):.0f}')
    ax4.axvline(x=np.mean(pred_x_vals) - np.std(pred_x_vals), color='orange', linestyle=':')
    ax4.axvline(x=np.mean(pred_x_vals) + np.std(pred_x_vals), color='orange', linestyle=':',
                label=f'±1σ')
    ax4.set_xlabel("X Position (pixels)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("X Distribution (Noise Profile)")
    ax4.legend()

    # ----- Plot 5: Y Coordinate Timeline -----
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.plot(pred_y_vals, label='Mapped Y', linewidth=0.8, alpha=0.7, color='orange')
    ax5.axhline(y=np.mean(pred_y_vals), color='red', linestyle='--',
                label=f'Mean: {np.mean(pred_y_vals):.0f}px')
    ax5.fill_between(range(len(pred_y_vals)),
                     np.mean(pred_y_vals) - np.std(pred_y_vals),
                     np.mean(pred_y_vals) + np.std(pred_y_vals),
                     alpha=0.2, color='orange', label=f'±1σ ({np.std(pred_y_vals):.1f}px)')
    ax5.set_xlabel("Frame")
    ax5.set_ylabel("Y Coordinate (pixels)")
    ax5.set_title(f"Vertical Gaze Timeline | SNR: {analysis['vertical']['snr']:.1f}:1")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right')

    # ----- Plot 6: Frame-to-frame Jitter -----
    ax6 = fig.add_subplot(gs[2, 2])
    dx = np.diff(pred_x_vals)
    dy = np.diff(pred_y_vals)
    jitter_magnitude = np.sqrt(dx**2 + dy**2)
    ax6.plot(jitter_magnitude, label='Jitter', color='red', linewidth=0.8, alpha=0.7)
    ax6.axhline(y=np.mean(jitter_magnitude), color='black', linestyle='--',
                label=f'Mean: {np.mean(jitter_magnitude):.1f}px')
    ax6.set_xlabel("Frame")
    ax6.set_ylabel("Distance (pixels)")
    ax6.set_title("Frame-to-Frame Jitter")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.suptitle("ADHD Gaze Tracker - SNR & Precision Analysis (4K Resolution)",
                 fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def main():
    # Get log file from command line or use default
    if len(sys.argv) > 1:
        FILE_PATH = sys.argv[1]
    else:
        # Find the most recent gaze_log or snr_analysis file
        logs = [f for f in os.listdir('.') if (f.startswith('gaze_log_') or f.startswith('snr_analysis_')) and f.endswith('.json')]
        if not logs:
            print("Error: No gaze_log or snr_analysis files found")
            return
        FILE_PATH = sorted(logs)[-1]

    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle different file formats
    log_entries = []
    raw_yaw_vals = []
    raw_pitch_vals = []

    if isinstance(data, list):
        log_entries = data
    elif "fixation_data" in data:
        # SNR analysis file format
        log_entries = data.get("fixation_data", [])
        print(f"Loading SNR analysis file with pre-computed results...")
        if "analysis" in data:
            print_snr_report(data["analysis"])
    elif "points" in data:
        log_entries = data.get("points", [])
    else:
        log_entries = [data]

    if not log_entries:
        print("Error: No data found in log file")
        return

    # Extract prediction data from log entries
    pred_x_vals = []
    pred_y_vals = []

    for entry in log_entries:
        # Check for various field names
        x_val = entry.get("target_x") or entry.get("mapped_x")
        y_val = entry.get("target_y") or entry.get("mapped_y") or entry.get("raw_target_y")

        if x_val is not None and y_val is not None:
            pred_x_vals.append(x_val)
            pred_y_vals.append(y_val)

        # Extract raw model outputs if available
        if "raw_yaw" in entry:
            raw_yaw_vals.append(entry["raw_yaw"])
        if "raw_pitch" in entry:
            raw_pitch_vals.append(entry["raw_pitch"])

    if not pred_x_vals:
        print("Error: No coordinate data found in log entries")
        return

    print(f"Analyzing {len(pred_x_vals)} frames from {FILE_PATH}\n")

    # Convert to numpy arrays
    pred_x_vals = np.array(pred_x_vals)
    pred_y_vals = np.array(pred_y_vals)
    raw_yaw = np.array(raw_yaw_vals) if raw_yaw_vals else None
    raw_pitch = np.array(raw_pitch_vals) if raw_pitch_vals else None

    # Perform SNR analysis
    analysis = analyze_snr_and_precision(
        pred_x_vals, pred_y_vals,
        raw_yaw, raw_pitch,
        SCREEN_WIDTH, SCREEN_HEIGHT
    )

    # Print detailed report
    print_snr_report(analysis)

    # Save analysis results
    analysis_output_path = FILE_PATH.replace('.json', '_analysis.json')
    with open(analysis_output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_output_path}")

    # Create visualization
    viz_output_path = FILE_PATH.replace('.json', '_snr_analysis.png')
    create_snr_visualization(pred_x_vals, pred_y_vals, analysis, save_path=viz_output_path)


if __name__ == "__main__":
    main()
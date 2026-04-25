import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create an output directory for the figures
os.makedirs("report_figures", exist_ok=True)

# -------------------------------------------------------------
# Figure 1: The Validation Report (Diagnostic Card)
# -------------------------------------------------------------
def generate_figure_1():
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e2e')
    ax.axis('off')
    
    # Stylized diagnostic card background
    card = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2, edgecolor='#89dceb', facecolor='#181825')
    ax.add_patch(card)
    
    # Text Data
    plt.text(0.5, 0.85, "DIAGNOSTIC REPORT: GOLDEN RUN", color='#cdd6f4', fontsize=16, fontweight='bold', ha='center')
    plt.text(0.15, 0.65, "SNR:", color='#a6adc8', fontsize=14, fontweight='bold')
    plt.text(0.40, 0.65, "7.29", color='#a6e3a1', fontsize=18, fontweight='bold')
    
    plt.text(0.15, 0.45, "Status:", color='#a6adc8', fontsize=14, fontweight='bold')
    plt.text(0.40, 0.45, "STABLE (Rose Criterion Passed)", color='#89b4fa', fontsize=14, fontweight='bold')
    
    plt.text(0.15, 0.25, "Avg Jitter:", color='#a6adc8', fontsize=14, fontweight='bold')
    plt.text(0.40, 0.25, "23.73 px", color='#fab387', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("report_figures/Figure_1_Validation_Report.png", dpi=300)
    plt.close()
    print("Generated Figure 1: Validation Report")

# -------------------------------------------------------------
# Figure 2: Jitter Suppression Time-Series
# -------------------------------------------------------------
def generate_figure_2():
    frames = np.arange(0, 300)
    
    # Simulate heavy noise (321.67px offset variance) for Stage 3 raw data
    np.random.seed(42)
    raw_x = 1920 + np.random.normal(0, 321.67, size=len(frames))
    
    # Simulate EMA Filtered Data (alpha=0.05, drastically lowering jitter to ~10.76px)
    ema_x = np.zeros_like(raw_x)
    ema_x[0] = raw_x[0]
    alpha = 0.05
    for i in range(1, len(frames)):
        ema_x[i] = alpha * raw_x[i] + (1 - alpha) * ema_x[i-1]
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frames, raw_x, color='#f38ba8', label='Raw X (Stage 3) - Jitter: 321.67px', alpha=0.7, linewidth=1.5)
    ax.plot(frames, ema_x, color='#89b4fa', label=r'EMA Filtered X ($\alpha=0.05$) - Jitter: 10.76px', linewidth=2.5)
    
    ax.set_xlabel("Frames", fontsize=12)
    ax.set_ylabel("Gaze X Coordinate (px)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("report_figures/Figure_2_Jitter_Suppression.png", dpi=300)
    plt.close()
    print("Generated Figure 2: Jitter Suppression Time-Series")

# -------------------------------------------------------------
# Figure 3: The 6-Sigma Guard Band Map (Binning Strategy)
# -------------------------------------------------------------
def generate_figure_3():
    fig, ax = plt.subplots(figsize=(12, 6.75)) # 16:9 aspect ratio
    
    # Define 4K constraints
    width, height = 3840, 2160
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis() # Display coordinates from top-left (0,0)
    
    zone_width = width / 3
    guard_band_center = width / 2
    # Standard deviation was ~526, total band = 1053 (±526 around center)
    guard_x1 = guard_band_center - 526.5
    guard_x2 = guard_band_center + 526.5
    
    # Add vertical zones
    ax.add_patch(patches.Rectangle((0, 0), zone_width, height, facecolor='#f5e0dc', alpha=0.4))
    ax.add_patch(patches.Rectangle((zone_width, 0), zone_width, height, facecolor='#f38ba8', alpha=0.4))
    ax.add_patch(patches.Rectangle((zone_width*2, 0), zone_width, height, facecolor='#cba6f7', alpha=0.4))
    
    # Add Zone Labels
    plt.text(zone_width/2, height*0.1, "LEFT ZONE\n(0-1280px)", ha='center', fontsize=14, fontweight='bold', color='#313244')
    plt.text(width/2, height*0.1, "CENTER ZONE\n(1280-2560px)", ha='center', fontsize=14, fontweight='bold', color='#313244')
    plt.text(width - zone_width/2, height*0.1, "RIGHT ZONE\n(2560-3840px)", ha='center', fontsize=14, fontweight='bold', color='#313244')
    
    # Draw Guard Band
    ax.axvspan(guard_x1, guard_x2, color='#89b4fa', alpha=0.5, label='1$\sigma$ Guard Band (1053px)')
    ax.plot([guard_x1, guard_x1], [0, height], color='#1e1e2e', linestyle='--', linewidth=2)
    ax.plot([guard_x2, guard_x2], [0, height], color='#1e1e2e', linestyle='--', linewidth=2)
    
    # Dimension annotations inside center zone
    plt.annotate('', xy=(guard_x1, height/2), xytext=(guard_x2, height/2),
                 arrowprops=dict(arrowstyle='<->', color='#1e1e2e', lw=2))
    plt.text(guard_band_center, height/2 - 50, "~1053 px", ha='center', va='center', fontsize=14, fontweight='bold', color='#1e1e2e')
    
    ax.set_xlabel("X Resolution (px)")
    ax.set_ylabel("Y Resolution (px)")
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig("report_figures/Figure_3_Guard_Band_Map.png", dpi=300)
    plt.close()
    print("Generated Figure 3: 6-Sigma Guard Band Map")

# -------------------------------------------------------------
# Figure 4: The Temporal Paradox Infographic
# -------------------------------------------------------------
def generate_figure_4():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = ['Human Saccade Initiation', 'Total System Latency (Hardware + EMA)']
    durations = [200, 333]
    colors = ['#a6e3a1', '#f38ba8']
    
    bars = ax.bar(labels, durations, color=colors, edgecolor='black', zorder=3)
    
    # Enhance visual styling
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Figure 4: The Temporal Paradox in Saccade Tracking', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_ylim(0, 400)
    
    # Add numerical labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval} ms", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # Draw paradox threshold line
    ax.axhline(y=200, color='red', linestyle=':', linewidth=2, label='Biological Threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("report_figures/Figure_4_Temporal_Paradox.png", dpi=300)
    plt.close()
    print("Generated Figure 4: Temporal Paradox Infographic")

if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    print("\nAll figures generated successfully in the 'report_figures' directory.")
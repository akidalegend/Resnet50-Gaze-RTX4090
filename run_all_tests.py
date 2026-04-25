#!/usr/bin/env python3
"""
AUTOMATED THREE-STAGE TEST RUNNER
Run all three benchmark tests sequentially without editing the script.

Usage:
    python run_all_tests.py           # Run all 3 stages (manual S press for each)
    python run_all_tests.py --auto    # Auto-run (requires editing test_stages.py)
"""

import subprocess
import sys
import shutil
from pathlib import Path
import json
from datetime import datetime
import time
import re

def run_stage(stage_num, input_mode, use_filter=True):
    """Run a single test stage."""
    print("\n" + "="*70)
    print(f"RUNNING STAGE {stage_num} - {input_mode} MODE (FILTER: {use_filter})")
    print("="*70)
    
    test_file = Path("test_stages.py")
    backup_file = Path("test_stages.py.bak")
    
    # Read the file
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Save backup of original
    with open(backup_file, 'w') as f:
        f.write(content)
    
    try:
        # Replace configuration using regex to handle comments
        # Match TEST_STAGE = <any number> (ignoring comments after)
        new_content = re.sub(
            r'TEST_STAGE = \d+',
            f'TEST_STAGE = {stage_num}',
            content
        )
        
        # Match INPUT_MODE = "<IMAGE|WEBCAM>" (ignoring comments after)
        new_content = re.sub(
            r'INPUT_MODE = "[^"]*"',
            f'INPUT_MODE = "{input_mode}"',
            new_content
        )
        
        # Match USE_FILTER
        new_content = re.sub(
            r'USE_FILTER = \w+',
            f'USE_FILTER = {use_filter}',
            new_content
        )
        
        # Write back
        with open(test_file, 'w') as f:
            f.write(new_content)
        
        print(f"✓ Configuration updated: TEST_STAGE={stage_num}, INPUT_MODE={input_mode}")
        
        # Verify the changes were applied
        with open(test_file, 'r') as f:
            verify = f.read()
            if f"TEST_STAGE = {stage_num}" in verify and f'INPUT_MODE = "{input_mode}"' in verify:
                print(f"✓ Verification successful: config correctly set")
            else:
                print(f"⚠ WARNING: Config verification failed")
        
        # Run the test
        result = subprocess.run([sys.executable, "test_stages.py"], check=False)
        
        return result.returncode == 0
    finally:
        # Restore backup AFTER test completes
        if backup_file.exists():
            shutil.move(backup_file, test_file)
            print(f"✓ Restored test_stages.py from backup")


def countdown_timer(seconds, label="Preparing", pre_message=""):
    """Display a countdown timer for stage preparation."""
    if pre_message:
        print(pre_message)
    
    print(f"\n{label}...\n")
    
    for remaining in range(seconds, 0, -1):
        minutes = remaining // 60
        secs = remaining % 60
        
        # Create visual bar
        bar_length = 40
        progress = (seconds - remaining) / seconds
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Display countdown
        print(f"\r  ⏳ {bar} {minutes:02d}:{secs:02d}  ", end="", flush=True)
        time.sleep(1)
    
    print(f"\r  ✓ Ready! (00:00)                                       \n")


def check_requirements():
    """Check if all required files exist."""
    required = [
        "test_stages.py",
        "calibration.json",
        "gaze_resnet50_epoch10.pth",
        "dataset/Data/Normalized/p00/p00-Normalized-0.jpg"
    ]
    
    missing = []
    for file in required:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("ERROR: Missing required files:")
        for file in missing:
            print(f"  ✗ {file}")
        return False
    
    print("✓ All required files found")
    return True


def summarize_results():
    """Read and summarize the three output JSON files."""
    import glob
    import json
    
    print("\n" + "="*70)
    print("SUMMARY OF ALL THREE STAGES")
    print("="*70)
    
    stages_data = {}
    for stage in [1, 2, 3, 4]:
        files = sorted(glob.glob(f"snr_analysis_stage{stage}_*.json"))
        if files:
            with open(files[-1], 'r') as f:
                data = json.load(f)
                stages_data[stage] = data['analysis']['horizontal']
        else:
            print(f"Warning: No analysis file found for Stage {stage}")
    
    print("\n{:<10} {:<15} {:<15} {:<15}".format("Stage", "Jitter (px)", "SNR", "Std Dev (px)"))
    print("-"*55)
    
    for stage in [1, 2, 3, 4]:
        if stage in stages_data:
            h_data = stages_data[stage]
            jitter = h_data['jitter_mean']
            snr = h_data['snr']
            sigma = h_data['std_sigma']
            print("{:<10} {:<15.2f} {:<15.2f} {:<15.2f}".format(stage, jitter, snr, sigma))
    
    print("\n" + "="*70)
    print("EXPECTED BEHAVIOR:")
    print("  Stage 1 → 2: Jitter ↑↑ (0.5 → 18), SNR ↓↓ (4500 → 25)  = Hardware culprit")
    print("  Stage 2 → 3: Jitter ↑ (18 → 24), SNR ↓ (25 → 7) but >5  = Filter recovers")
    print("="*70 + "\n")
    
    # Check if Stage 3 passes Rose Criterion
    if 3 in stages_data:
        snr_3 = stages_data[3]['snr']
        if snr_3 >= 5.0:
            print("✓ STAGE 3 PASSES ROSE CRITERION (SNR >= 5.0) ✓")
        else:
            print("✗ Stage 3 SNR below threshold - tuning needed")


def main():
    print("\n" + "="*70)
    print("THREE-STAGE GAZE SYSTEM TEST RUNNER")
    print("="*70)
    
    # Check requirements
    if not check_requirements():
        print("\nFix missing files and try again.")
        sys.exit(1)
    
    print("\nTest Configuration:")
    print("  Stage 1: IMAGE mode (static normalized face) [Filter ON]")
    print("  Stage 2: WEBCAM mode (printed photo on stand) [Filter ON]")
    print("  Stage 3: WEBCAM mode (you as subject with chin rest) [Filter OFF]")
    print("  Stage 4: WEBCAM mode (you as subject with chin rest) [Filter ON]")
    
    print("\nFor each stage:")
    print("  1. Window appears showing input source")
    print("  2. Press 'S' to START collecting data")
    print("  3. Run for 300 frames (~10 seconds)")
    print("  4. Auto-analysis after completion")
    print("  5. JSON file saved, then next stage starts")
    
    input("\n► Press ENTER to start Stage 1 (static image test)...\n")
    if not run_stage(1, "IMAGE", True):
        print("Stage 1 failed!")
        sys.exit(1)
    
    countdown_timer(
        15,
        "Preparing for Stage 2",
        "\n" + "="*70 +
        "\n► Stage 1 complete! Preparing Stage 2...\n" +
        "⚠️  SETUP FOR STAGE 2:\n" +
        "   • Place printed face photo on stand in front of webcam\n" +
        "   • Ensure photo is NOT held with hands (to avoid shaking)\n" +
        "   • Adjust lighting to match Stage 1 brightness\n" +
        "="*70
    )
    if not run_stage(2, "WEBCAM", True):
        print("Stage 2 failed!")
        sys.exit(1)
    
    countdown_timer(
        20,
        "Preparing for Stage 3",
        "\n" + "="*70 +
        "\n► Stage 2 complete! Preparing Stage 3...\n" +
        "⚠️  SETUP FOR STAGE 3:\n" +
        "   • Build chin rest (stack of heavy books)\n" +
        "   • Position so your head cannot move\n" +
        "   • Prepare to stare straight ahead for ~10 seconds\n" +
        "   • Ensure similar lighting to previous stages\n" +
        "="*70
    )
    if not run_stage(3, "WEBCAM", False):
        print("Stage 3 failed!")
        sys.exit(1)
    
    countdown_timer(
        15,
        "Preparing for Stage 4",
        "\n" + "="*70 +
        "\n► Stage 3 complete! Preparing Stage 4...\n" +
        "⚠️ SETUP FOR STAGE 4:\n" +
        "   • Stay in exactly the same position (chin rest)\n" +
        "   • Filter is now turning ON to measure improvement\n" +
        "="*70
    )
    if not run_stage(4, "WEBCAM", True):
        print("Stage 4 failed!")
        sys.exit(1)
        
    # Summarize results
    print("\n✓ All stages complete!")
    summarize_results()
    
    print("\n► Test results saved. Review:")
    print("   - snr_analysis_stage1_*.json")
    print("   - snr_analysis_stage2_*.json")
    print("   - snr_analysis_stage3_*.json")
    print("\n► For detailed methodology, see TESTING_PROTOCOL.md")


if __name__ == "__main__":
    main()

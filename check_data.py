import json
import glob

files = sorted(glob.glob('snr_analysis_stage*.json'))[-3:]
for file in files:
    with open(file) as f:
        data = json.load(f)
        h = data['analysis']['horizontal']
        print(f'\n{file}:')
        print(f'  Jitter: {h["jitter_mean"]}')
        print(f'  StdDev: {h["std_sigma"]}')
        print(f'  SNR: {h["snr"]}')
        print(f'  Range: {h["range"]}')
        print(f'  Status: {h["status"]}')
        
        # Show first few data points
        if data.get('fixation_data'):
            print(f'  Sample data points:')
            for i, pt in enumerate(data['fixation_data'][:3]):
                print(f'    [{i}] mapped_x={pt.get("mapped_x")}, mapped_y={pt.get("mapped_y")}, raw_yaw={pt.get("raw_yaw")}')

import os
import pandas as pd

# Path to where you extracted the data
# Based on your image, it should be:
base_dir = "./dataset/Data/Original"
output_file = "./dataset/labels.csv"

data_list = []

if not os.path.exists(base_dir):
    print(f"ERROR: Could not find folder: {base_dir}")
    print(f"Check if 'Data' and 'Original' are capitalized correctly.")
else:
    print(f"Success: Found {base_dir}. Scanning for participants...")

    # Loop through p00, p01, etc.
    participants = [d for d in os.listdir(base_dir) if d.startswith('p')]
    print(f"Found {len(participants)} participant folders: {participants}")

    for person in participants:
        person_path = os.path.join(base_dir, person)
        
        # Each person has 'dayXX' folders
        days = [d for d in os.listdir(person_path) if os.path.isdir(os.path.join(person_path, d))]
        
        for day in days:
            day_path = os.path.join(person_path, day)
            # The label file is usually named 'dayXX.txt' or 'annotation.txt'
            # Let's look for ANY .txt file in that folder
            txt_files = [f for f in os.listdir(day_path) if f.endswith('.txt')]
            
            for txt_file in txt_files:
                annotation_path = os.path.join(day_path, txt_file)
                with open(annotation_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 3: continue
                        
                        # Store path relative to the 'dataset' folder
                        img_rel_path = f"Data/Original/{person}/{day}/{parts[0]}"
                        # Pitch and Yaw
                        pitch = parts[1]
                        yaw = parts[2]
                        
                        data_list.append([img_rel_path, pitch, yaw])

    # Save to CSV
    if len(data_list) > 0:
        df = pd.DataFrame(data_list, columns=['image_path', 'pitch', 'yaw'])
        df.to_csv(output_file, index=False)
        print(f"DONE! Created {output_file} with {len(df)} labels.")
    else:
        print("STILL 0 LABELS: Check if there are .txt files inside the 'day' folders.")
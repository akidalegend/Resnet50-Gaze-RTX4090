import pandas as pd
import os
from tqdm import tqdm

print("Loading original labels...")
# Read the messy CSV
df = pd.read_csv("./dataset/labels.csv")
valid_rows = []

# Your exact dataset folder
root_dir = r"C:\Users\23828140\Documents\GitHub\Resnet50-Gaze-RTX4090\dataset"

print("Verifying which images actually exist on your SSD. This will take a minute...")
for index, row in tqdm(df.iterrows(), total=len(df)):
    raw_path = str(row.iloc[0])
    
    # Extract the parts
    parts = raw_path.replace("\\", "/").split("/")
    try:
        p_index = [i for i, s in enumerate(parts) if s.startswith('p') and len(s) == 3][0]
        person = parts[p_index]
        day = parts[p_index + 1]
        
        # Apply the 4-digit zero padding rule you discovered!
        filename = parts[p_index + 2].split('.')[0].zfill(4) + ".jpg"
        
        # Build the exact path
        img_path = os.path.join(root_dir, "Data", "Original", person, day, filename)
        
        # The Golden Rule: Only keep it if the file is physically there
        if os.path.exists(img_path):
            valid_rows.append({
                'image_path': img_path,  # We save the PERFECT path
                'pitch': row.iloc[1], 
                'yaw': row.iloc[2]
            })
    except:
        continue

clean_df = pd.DataFrame(valid_rows)
clean_df.to_csv("./dataset/clean_labels.csv", index=False)
print(f"\n✅ Data Cleaned! Kept {len(clean_df)} valid images out of {len(df)}.")
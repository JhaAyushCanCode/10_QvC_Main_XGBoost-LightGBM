# prepare_data.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = r"C:\Users\Admin\.spyder-py3\QvC-3_docs\full_dataset"
OUTPUT_DIR = r"C:\Users\Admin\.spyder-py3\QvC-3_docs"

# Files
files = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

# Load and combine
dfs = []
for f in files:
    path = os.path.join(DATA_DIR, f)
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f" Combined dataset size: {len(data)}")

# Select text + 28 emotions
label_cols = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval','disgust',
    'embarrassment','excitement','fear','gratitude','grief','joy','love',
    'nervousness','optimism','pride','realization','relief','remorse','sadness',
    'surprise','neutral'
]

data = data[['text'] + label_cols]

# Convert label columns into list of ints
data['labels'] = data[label_cols].values.tolist()

# Drop the original individual label columns
data = data[['text', 'labels']]

# Split into train/val/test
train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save
train_path = os.path.join(OUTPUT_DIR, "train.csv")
val_path   = os.path.join(OUTPUT_DIR, "val.csv")
test_path  = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f" Saved train/val/test splits to {OUTPUT_DIR}")
print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


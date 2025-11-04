# 03_IMAGE_PREPROCESSING — FINAL STABLE RESUMABLE VERSION (FIXED)

import os, cv2, numpy as np, pandas as pd, concurrent.futures
from tqdm import tqdm

# --- 1. Config ---
IMG_SIZE = 299  # The size for Xception

# --- 2. Base Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_preprocessed")
# This is the folder inside 'data_raw' that holds the images
IMAGE_DIR = os.path.join(INPUT_DIR, "real_vs_fake", "real-vs-fake")

# Define all destination directories
train_real_dir = os.path.join(OUTPUT_DIR, "train", "real")
train_fake_dir = os.path.join(OUTPUT_DIR, "train", "fake")
val_real_dir = os.path.join(OUTPUT_DIR, "validation", "real")
val_fake_dir = os.path.join(OUTPUT_DIR, "validation", "fake")

# Create directories upfront
os.makedirs(train_real_dir, exist_ok=True)
os.makedirs(train_fake_dir, exist_ok=True)
os.makedirs(val_real_dir, exist_ok=True)
os.makedirs(val_fake_dir, exist_ok=True)

# --- 3. Load Metadata ---
try:
    train_csv_path = os.path.join(INPUT_DIR, "train.csv")
    valid_csv_path = os.path.join(INPUT_DIR, "valid.csv")
    
    df_train = pd.read_csv(train_csv_path)
    df_valid = pd.read_csv(valid_csv_path)
except FileNotFoundError:
    print(f"❌ ERROR: Could not find train.csv or valid.csv at path: {INPUT_DIR}")
    print("Please make sure 'data_raw' contains the unzipped Kaggle files.")
    exit()

# Combine them into one big dataframe 'df'
df_train['split'] = 'train'
df_valid['split'] = 'validation'
df = pd.concat([df_train, df_valid])
df['label'] = df['label'].apply(lambda x: "real" if x == 1 else "fake")

# --- ‼️ CRITICAL FIX HERE ‼️ ---
# This line now correctly splits the path (e.g., "valid/fake/img.jpg")
# and joins it using the correct Windows '\' slashes.
df['image_path'] = df['path'].apply(lambda x: os.path.join(IMAGE_DIR, *x.split('/')))
# --- ‼️ END OF FIX ‼️ ---

print(f"Found {len(df):,} total images to process.")


# --- 4. Resume checkpoint if exists ---
log_path = os.path.join(OUTPUT_DIR, "preprocess_log.csv")
done_files = set()
if os.path.exists(log_path):
    try:
        done_files = set(pd.read_csv(log_path)["dest_path"])
        print(f"🔁 Found checkpoint: {len(done_files):,} already processed.")
    except Exception as e:
        print(f"⚠ Couldn't read checkpoint, starting fresh. Error: {e}")

# --- 5. Core preprocessing (Resize-Only) ---
def preprocess_and_save(image_path, dest_path):
    """
    FIXED: This function now ONLY resizes and saves.
    """
    if dest_path in done_files or os.path.exists(dest_path):
        return (False, "skipped") # Return False, reason

    try:
        img = cv2.imread(image_path)
        if img is None:
            # This error will now be correct, the path was checked
            return (False, "error_read") 

        # Only resize if it's not already the correct size
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(dest_path, img)
        return (dest_path, "processed") # Return dest_path, reason

    except Exception as e:
        return (False, "error_exception")


# --- 6. Process Row Function ---
def process_row(row):
    """
    Takes a row from the DataFrame and calls preprocess_and_save.
    'row' is a dictionary.
    """
    src = row['image_path'] # Full source path from our 'df'
    split = row['split']
    label = row['label']
    
    if label == 'real':
        dest_dir = train_real_dir if split == 'train' else val_real_dir
    else:
        dest_dir = train_fake_dir if split == 'train' else val_fake_dir
        
    # The 'path' column (e.g., valid/fake/img.jpg) is not just the filename
    # We need to get the filename itself
    fname = os.path.basename(src) # e.g., 'img.jpg'
    dest_path = os.path.join(dest_dir, fname)
    
    return process_row_result(preprocess_and_save(src, dest_path), dest_path)

def process_row_result(result, dest_path):
    """Helper to handle the tuple result from preprocess_and_save"""
    path_or_false, reason = result
    if reason == "processed":
        return dest_path, "processed"
    return False, reason


# --- 7. Main loop ---
print("🚀 Starting threaded preprocessing (Resize-only)...")

new_done_paths = []
processed_count = 0
skipped_count = 0
error_count = 0
BATCH_SIZE = 5000

# Convert DataFrame to list of dictionaries for faster processing
jobs = df.to_dict("records")

for i in tqdm(range(0, len(jobs), BATCH_SIZE), desc="🧠 Batch Processing", ncols=90):
    batch_jobs = jobs[i:i+BATCH_SIZE]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_row, batch_jobs))

    # Process results for this batch
    batch_new_paths = []
    for dest_path, reason in results:
        if reason == "processed":
            processed_count += 1
            if dest_path: 
                batch_new_paths.append(dest_path)
        elif reason == "skipped":
            skipped_count += 1
        else: # "error_read" or "error_exception"
            error_count += 1
            
    new_done_paths.extend(batch_new_paths)

    # Save progress checkpoint
    if batch_new_paths:
        pd.DataFrame(batch_new_paths, columns=["dest_path"]).to_csv(
            log_path, mode="a", index=False, header=not os.path.exists(log_path)
        )

print("\n✅ Preprocessing complete.")
print(f"🖼  {processed_count:,} new images processed and saved to '{OUTPUT_DIR}'.")
print(f"⏭  {skipped_count + len(done_files):,} images skipped (already existed).")
print(f"❌ {error_count:,} images failed to process.")
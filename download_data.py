import os, sys, subprocess, zipfile
from tqdm import tqdm

# --- Settings ---
DATASET_NAME = "xhlulu/140k-real-and-fake-faces"
DOWNLOAD_PATH = "."
ZIP_FILE_NAME = "140k-real-and-fake-faces.zip"
EXTRACT_FOLDER = "raw_data"
# ----------------

def download_dataset():
    print(f"📥 Downloading dataset: {DATASET_NAME}")
    print("⏳ Real-time download progress below:\n")

    cmd = f'kaggle datasets download -d {DATASET_NAME} -p {DOWNLOAD_PATH} --force'
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

    for line in iter(process.stdout.readline, ""):
        if line.strip():
            print(line, end="")
        sys.stdout.flush()

    process.wait()

    if process.returncode != 0 or not os.path.exists(ZIP_FILE_NAME):
        raise Exception("❌ Download failed")

    print("\n✅ Download Complete")

def extract_zip():
    print(f"📦 Extracting into '{EXTRACT_FOLDER}' ...")
    
    with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as z:
        members = z.infolist()
        with tqdm(total=len(members), desc="Extracting", unit="files") as pbar:
            for m in members:
                z.extract(m, EXTRACT_FOLDER)
                pbar.update(1)

    print("🧹 Removing ZIP file...")
    os.remove(ZIP_FILE_NAME)
    print("✅ Extraction Finished")

if __name__ == "__main__":
    download_dataset()
    extract_zip()
    print("\n🎉 Dataset ready!")

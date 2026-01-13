import os
import shutil
import random

# --- CONFIGURARE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')

def main():
    print("📦 Se împarte dataset-ul (70% Train, 15% Val, 15% Test)...")
    
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in ['liber', 'ocupat']:
            os.makedirs(os.path.join(folder, cls), exist_ok=True)

    for cls in ['liber', 'ocupat']:
        src_path = os.path.join(SOURCE_DIR, cls)
        files = os.listdir(src_path)
        random.shuffle(files)

        n = len(files)
        tr = int(n * 0.7)
        vl = int(n * 0.85)

        # Split
        train_files = files[:tr]
        val_files = files[tr:vl]
        test_files = files[vl:]

        print(f"Clasa {cls.upper()}: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")

        # Mutare fișiere
        for f in train_files: shutil.copy2(os.path.join(src_path, f), os.path.join(TRAIN_DIR, cls, f))
        for f in val_files:   shutil.copy2(os.path.join(src_path, f), os.path.join(VAL_DIR, cls, f))
        for f in test_files:  shutil.copy2(os.path.join(src_path, f), os.path.join(TEST_DIR, cls, f))

    print("✅ Dataset-ul a fost structurat pentru antrenare!")

if __name__ == "__main__":
    main()
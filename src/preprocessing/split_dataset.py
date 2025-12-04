import os
import shutil
import random
import glob

# --- CONFIGURARE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DEST_DIR = os.path.join(BASE_DIR, 'data')

# Procente pentru împărțire (Total = 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def clear_folder(folder_path):
    """Șterge conținutul folderului pentru a nu amesteca date vechi."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def copy_files(files, source, destination, category):
    """Copiază o listă de fișiere în destinație."""
    dest_path = os.path.join(destination, category)
    os.makedirs(dest_path, exist_ok=True)
    
    for f in files:
        shutil.copy2(os.path.join(source, category, f), os.path.join(dest_path, f))

def main():
    print(">>> Începere împărțire dataset (Train / Val / Test)...")

    # 1. Curățăm folderele destinație
    for split in ['train', 'validation', 'test']:
        clear_folder(os.path.join(DEST_DIR, split))

    # 2. Procesăm fiecare clasă (liber / ocupat)
    for category in ['liber', 'ocupat']:
        src_path = os.path.join(SOURCE_DIR, category)
        if not os.path.exists(src_path):
            print(f"Eroare: Nu găsesc folderul {src_path}")
            continue

        # Luăm toate imaginile
        images = os.listdir(src_path)
        random.shuffle(images) # Le amestecăm bine
        
        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        
        # Slicing (tăiem lista)
        train_files = images[:train_count]
        val_files = images[train_count : train_count + val_count]
        test_files = images[train_count + val_count :]

        print(f"Clasa '{category}': Total {total} -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        # Copiem fizic fișierele
        copy_files(train_files, SOURCE_DIR, os.path.join(DEST_DIR, 'train'), category)
        copy_files(val_files, SOURCE_DIR, os.path.join(DEST_DIR, 'validation'), category)
        copy_files(test_files, SOURCE_DIR, os.path.join(DEST_DIR, 'test'), category)

    print(">>> Gata! Folderele data/train, data/validation și data/test sunt populate.")

if __name__ == "__main__":
    main()
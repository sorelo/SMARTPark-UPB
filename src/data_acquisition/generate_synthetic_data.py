import cv2
import numpy as np
import json
import os
import glob
import random

# Definire cai radacina
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BG_DIR = os.path.join(BASE_DIR, 'data', 'assets', 'backgrounds')
CAR_DIR = os.path.join(BASE_DIR, 'data', 'assets', 'cars')
CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'parking_spots.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'generated')

# Numar de variante per scenariu
IMAGES_PER_SCENARIO = 100 

def rotate_image(image, angle):
    """Roteste imaginea fara a taia marginile."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def overlay_transparent(bg, overlay, x, y, w, h):
    """Suprapune o imagine PNG cu transparenta peste fundal."""
    overlay_r = cv2.resize(overlay, (w, h))
    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg
    
    alpha = overlay_r[:, :, 3] / 255.0
    roi = bg[y:y+h, x:x+w]
    
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - alpha) + overlay_r[:, :, c] * alpha
    bg[y:y+h, x:x+w] = roi
    return bg

def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"Eroare: Fisierul de configurare {CONFIG_FILE} nu exista.")
        return

    for cat in ['liber', 'ocupat']:
        os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    
    car_paths = glob.glob(os.path.join(CAR_DIR, '*.png'))
    cars = [cv2.imread(c, -1) for c in car_paths]
    
    if not cars:
        print("Eroare: Nu am gasit imagini PNG cu masini in data/assets/cars.")
        return

    print(f"Incep generarea folosind {os.path.basename(CONFIG_FILE)}...")
    total_count = 0
    
    for bg_name, spots in configs.items():
        bg_path = os.path.join(BG_DIR, bg_name)
        base_img = cv2.imread(bg_path)
        if base_img is None:
            continue

        for period in range(3): # 0: Dimineata, 1: Pranz, 2: Seara
            for s_idx in range(IMAGES_PER_SCENARIO):
                scene = base_img.copy()
                current_labels = []

                for spot in spots:
                    is_occupied = random.random() > 0.5
                    poly = np.array(spot, dtype=np.int32)
                    rect = cv2.boundingRect(poly)
                    x, y, w, h = rect
                    
                    if is_occupied:
                        car = random.choice(cars)
                        angle = np.degrees(np.arctan2(spot[1][1] - spot[0][1], spot[1][0] - spot[0][0]))
                        scale = random.uniform(0.85, 0.98)
                        nw, nh = int(w * scale), int(h * scale)
                        nx, ny = x + (w - nw) // 2, y + (h - nh) // 2
                        scene = overlay_transparent(scene, rotate_image(car, angle), nx, ny, nw, nh)
                    
                    current_labels.append((rect, is_occupied))

                if period == 1: 
                    scene = cv2.convertScaleAbs(scene, alpha=1.2, beta=10)
                elif period == 2: 
                    scene = cv2.convertScaleAbs(scene, alpha=0.6, beta=-30)

                for (rect, occ) in current_labels:
                    rx, ry, rw, rh = rect
                    if rw <= 0 or rh <= 0: continue
                    crop = scene[ry:ry+rh, rx:rx+rw]
                    if crop.size == 0: continue
                    
                    final_roi = cv2.resize(crop, (64, 64))
                    category = 'ocupat' if occ else 'liber'
                    file_name = f"img_{total_count}_{period}_{s_idx}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, category, file_name), final_roi)
                    total_count += 1

    print(f"Succes. Au fost generate {total_count} imagini in {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()
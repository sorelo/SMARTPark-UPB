import cv2
import numpy as np
import json
import os
import glob
import random

# --- CONFIGURARE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(BASE_DIR, 'data', 'assets')
BG_DIR = os.path.join(ASSETS_DIR, 'backgrounds')
CAR_DIR = os.path.join(ASSETS_DIR, 'cars')
CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'parking_spots.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Vrem ~3600 de imagini total (ex: 4 fundaluri * 3 perioade * 30 scenarii * locuri_per_parcare)
IMAGES_PER_SCENARIO = 30 

def rotate_image(image, angle):
    """Rotește imaginea păstrând transparența (canalul Alpha)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Folosim borderValue=(0,0,0,0) pentru a păstra transparența la margini
    result = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return result

def overlay_transparent(background, overlay, x, y, target_w, target_h):
    """Suprapune o mașină PNG peste fundal în coordonatele specificate."""
    overlay_resized = cv2.resize(overlay, (target_w, target_h))
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay_resized.shape

    # Verificare limite
    if x + w > bg_w or y + h > bg_h or x < 0 or y < 0:
        return background

    # Extragem canalele
    b, g, r, a = cv2.split(overlay_resized)
    overlay_color = cv2.merge((b, g, r))
    mask = a / 255.0

    # Aplicăm masca pentru blending
    roi = background[y:y+h, x:x+w]
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - mask) + overlay_color[:, :, c] * mask

    background[y:y+h, x:x+w] = roi
    return background

def apply_environment(image, period):
    """Simulează condițiile de iluminare (0: Dimineața, 1: Prânz, 2: Seară)."""
    if period == 0: # Dimineața (Neutral)
        return cv2.convertScaleAbs(image, alpha=1.0, beta=5)
    elif period == 1: # Prânz (Bright)
        return cv2.convertScaleAbs(image, alpha=1.2, beta=15)
    else: # Seara (Dark/Blue tint)
        image = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)
        overlay = np.full(image.shape, (50, 20, 10), dtype='uint8') # Albastru închis
        return cv2.addWeighted(image, 0.8, overlay, 0.2, 0)

def main():
    if not os.path.exists(CONFIG_FILE):
        print("Eroare: Nu am găsit parking_spots.json! Trasează locurile mai întâi.")
        return

    with open(CONFIG_FILE, 'r') as f:
        configs = json.load(f)

    car_files = glob.glob(os.path.join(CAR_DIR, "*.png"))
    if not car_files:
        print(f"Eroare: Nu am găsit mașini PNG în {CAR_DIR}")
        return

    loaded_cars = [cv2.imread(c, cv2.IMREAD_UNCHANGED) for c in car_files]
    
    # Pregătim folderele de ieșire
    os.makedirs(os.path.join(OUTPUT_DIR, 'liber'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'ocupat'), exist_ok=True)

    count = 0
    print("🚀 Se generează dataset-ul sintetic...")

    for bg_name, spots in configs.items():
        bg_path = os.path.join(BG_DIR, bg_name)
        base_img = cv2.imread(bg_path)
        if base_img is None: continue

        for period in range(3): # 3 momente ale zilei
            for i in range(IMAGES_PER_SCENARIO):
                scene = base_img.copy()
                decisions = []

                # Pasul 1: Construim scena completă (plasăm mașini aleatoriu)
                for spot in spots:
                    poly = np.array(spot, dtype=np.int32)
                    rect = cv2.boundingRect(poly)
                    x, y, w, h = rect
                    
                    is_occupied = random.random() > 0.5 # 50% șansă ocupare
                    if is_occupied:
                        car = random.choice(loaded_cars)
                        # Calculăm unghiul locului (folosind primele două puncte)
                        angle = np.degrees(np.arctan2(spot[1][1]-spot[0][1], spot[1][0]-spot[0][0]))
                        rotated_car = rotate_image(car, angle)
                        
                        # Scalare aleatorie (mașina să nu umple perfect locul)
                        scale = random.uniform(0.85, 0.95)
                        tw, th = int(w * scale), int(h * scale)
                        ox, oy = x + (w-tw)//2, y + (h-th)//2
                        
                        scene = overlay_transparent(scene, rotated_car, ox, oy, tw, th)
                    
                    decisions.append((rect, is_occupied))

                # Pasul 2: Aplicăm efectele de mediu pe toată scena
                final_scene = apply_environment(scene, period)

                # Pasul 3: Decupăm fiecare loc individual pentru dataset-ul CNN
                for rect, occ in decisions:
                    rx, ry, rw, rh = rect
                    crop = final_scene[ry:ry+rh, rx:rx+rw]
                    if crop.size == 0: continue
                    
                    crop_resized = cv2.resize(crop, (64, 64))
                    label = "ocupat" if occ else "liber"
                    filename = f"synth_{count}_{label}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, label, filename), crop_resized)
                    count += 1

    print(f"Gata! S-au generat {count} imagini în data/processed/")

if __name__ == "__main__":
    main()
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
CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'synthetic_spots.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Câte imagini generate vrei per fundal, per interval orar?
IMAGES_PER_SCENARIO = 5 

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return result

def overlay_transparent(background, overlay, x, y, target_w, target_h):
    # Redimensionare mașină
    overlay_resized = cv2.resize(overlay, (target_w, target_h))
    
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay_resized.shape

    if x + w > bg_w or y + h > bg_h or x < 0 or y < 0:
        return background

    # Separare canale (BGR + Alpha)
    if overlay_resized.shape[2] < 4:
        return background # Nu are alpha

    b, g, r, a = cv2.split(overlay_resized)
    overlay_rgb = cv2.merge((b, g, r))
    mask = a / 255.0

    # Zona de pe fundal unde punem mașina
    roi = background[y:y+h, x:x+w]

    # Combinare
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - mask) + overlay_rgb[:, :, c] * mask

    background[y:y+h, x:x+w] = roi
    return background

def apply_time_of_day(image, period):
    # period: 0 (8-12), 1 (12-16), 2 (16-20)
    
    if period == 0: # Dimineața (Lumină neutră, ușor rece)
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=10)
    
    elif period == 1: # Prânz (Lumină puternică, contrast mare)
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
    
    elif period == 2: # Seară (Mai întunecat, tentă portocalie/albastră)
        # Scădem luminozitatea
        image = cv2.convertScaleAbs(image, alpha=0.7, beta=-10)
        # Adăugăm o tentă de albastru/portocaliu (artificial)
        overlay = np.full(image.shape, (20, 10, 60), dtype='uint8') # Tentă albăstruie
        image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
        
    return image

def main():
    if not os.path.exists(CONFIG_FILE):
        print("Rulează întâi config_backgrounds.py!")
        return

    with open(CONFIG_FILE, 'r') as f:
        configs = json.load(f)

    car_files = glob.glob(os.path.join(CAR_DIR, "*.png"))
    if not car_files:
        print(f"Nu am găsit mașini PNG în {CAR_DIR}")
        return

    # Încărcăm mașinile în memorie (păstrăm canalul Alpha)
    loaded_cars = [cv2.imread(c, cv2.IMREAD_UNCHANGED) for c in car_files]

    os.makedirs(os.path.join(OUTPUT_DIR, 'liber'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'ocupat'), exist_ok=True)

    global_counter = 0

    # Iterăm prin fiecare fundal
    for bg_name, spots in configs.items():
        bg_path = os.path.join(BG_DIR, bg_name)
        base_img = cv2.imread(bg_path)
        if base_img is None: continue

        print(f"Procesare fundal: {bg_name}...")

        # Iterăm prin cele 3 perioade ale zilei
        for period in range(3): # 0, 1, 2
            period_name = ["Dimineata", "Pranz", "Seara"][period]
            
            # Generăm N variații pentru acest scenariu
            for i in range(IMAGES_PER_SCENARIO):
                # Copie proaspătă a fundalului
                scenario_img = base_img.copy()
                
                # Decidem aleatoriu ce locuri sunt ocupate
                # Seara (period 2) poate e mai liber, Prânz (period 1) mai plin
                occupancy_rate = 0.8 if period == 1 else 0.5
                
                spots_status = [] # Stocăm starea pentru decupare ulterioară

                for spot in spots:
                    # Calculăm bounding box-ul locului
                    poly = np.array(spot, dtype=np.int32)
                    rect = cv2.boundingRect(poly)
                    x, y, w, h = rect
                    
                    is_occupied = random.random() < occupancy_rate
                    
                    if is_occupied:
                        # Alegem o mașină random
                        car_img = random.choice(loaded_cars)
                        
                        # Rotim mașina (calculăm unghiul locului de parcare)
                        # Simplificare: calculăm unghiul primei laturi a poligonului
                        dx = spot[1][0] - spot[0][0]
                        dy = spot[1][1] - spot[0][1]
                        angle = np.degrees(np.arctan2(dy, dx))
                        
                        # Rotire și suprapunere
                        rotated_car = rotate_image(car_img, angle)
                        
                        # Potrivire dimensiuni (cu puțină variație aleatoare)
                        scale_factor = random.uniform(0.8, 1.0)
                        target_w = int(w * scale_factor)
                        target_h = int(h * scale_factor)
                        
                        # Centrare
                        offset_x = x + (w - target_w)//2
                        offset_y = y + (h - target_h)//2

                        scenario_img = overlay_transparent(scenario_img, rotated_car, offset_x, offset_y, target_w, target_h)
                        spots_status.append((spot, True)) # Spot, Ocupat
                    else:
                        spots_status.append((spot, False)) # Spot, Liber

                # Aplicăm efectul de timp (lumină/culoare) PE TOATĂ imaginea (fundal + mașini)
                final_img = apply_time_of_day(scenario_img, period)

                # --- DECUPARE ȘI SALVARE ---
                for idx, (spot, status) in enumerate(spots_status):
                    poly = np.array(spot, dtype=np.int32)
                    rect = cv2.boundingRect(poly)
                    x, y, w, h = rect
                    
                    # Decupăm locul
                    crop = final_img[y:y+h, x:x+w]
                    
                    if crop.size == 0: continue

                    label = "ocupat" if status else "liber"
                    filename = f"syn_{bg_name}_{period_name}_{i}_{idx}_{global_counter}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, label, filename)
                    
                    # Redimensionăm la 64x64 sau similar pentru CNN (opțional, dar recomandat)
                    crop_resized = cv2.resize(crop, (128, 128))
                    
                    cv2.imwrite(save_path, crop_resized)
                    global_counter += 1

    print(f"Gata! Am generat {global_counter} imagini în data/processed/")

if __name__ == "__main__":
    main()
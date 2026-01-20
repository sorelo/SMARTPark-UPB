import cv2
import torch
import numpy as np
import json
import os
import glob
import random
import sys
from PIL import Image
from torchvision import transforms

# Configurare cai
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.neural_network.model import ParkingCNN

ASSETS_DIR = os.path.join(BASE_DIR, 'data', 'assets')
BG_DIR = os.path.join(ASSETS_DIR, 'backgrounds')
CAR_DIR = os.path.join(ASSETS_DIR, 'cars')
CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'parking_spots.json')

def rotate_image(image, angle):
    """Functie auxiliara pentru rotirea masinilor in simulare."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return result

def overlay_transparent(background, overlay, x, y, target_w, target_h):
    """Suprapunere PNG cu transparenta."""
    overlay_resized = cv2.resize(overlay, (target_w, target_h))
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay_resized.shape

    if x + w > bg_w or y + h > bg_h or x < 0 or y < 0: return background
    if overlay_resized.shape[2] < 4: return background

    b, g, r, a = cv2.split(overlay_resized)
    overlay_rgb = cv2.merge((b, g, r))
    mask = a / 255.0
    
    roi = background[y:y+h, x:x+w]
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - mask) + overlay_rgb[:, :, c] * mask
    background[y:y+h, x:x+w] = roi
    return background

def main():
    # 1. Incarcare Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pornire Demo pe: {device}")
    
    model = ParkingCNN().to(device)
    
    # --- LOGICA DE CAUTARE MODEL (Actualizata) ---
    possible_paths = [
        os.path.join(BASE_DIR, 'models', 'trained_model.pth'),
        os.path.join(BASE_DIR, 'models', 'trained_model.h5'),
        os.path.join(BASE_DIR, 'data', 'parking_model.pth')
    ]
    
    load_path = None
    for p in possible_paths:
        if os.path.exists(p):
            load_path = p
            break
    
    if load_path:
        print(f"Încărcare model din: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
        print("Model incarcat cu succes.")
    else:
        print("Eroare: Nu gasesc modelul antrenat!")
        print(f"Am căutat în: {possible_paths}")
        return
        
    model.eval()

    # Transformari pentru inferenta
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Incarcare Resurse
    if not os.path.exists(CONFIG_FILE):
        print("Eroare: Nu exista configurarea locurilor.")
        return

    with open(CONFIG_FILE, 'r') as f:
        configs = json.load(f)
    
    bg_names = list(configs.keys())
    if not bg_names:
        print("Config empty.")
        return

    car_files = glob.glob(os.path.join(CAR_DIR, "*.png"))
    loaded_cars = [cv2.imread(c, -1) for c in car_files]

    # Stare initiala
    current_bg_idx = 0
    
    def get_current_scene_data(idx):
        fname = bg_names[idx]
        img_p = os.path.join(BG_DIR, fname)
        img = cv2.imread(img_p)
        if img is None:
            print(f"Atenție: Nu am putut încărca fundalul {img_p}")
            return np.zeros((720, 1280, 3), dtype=np.uint8), [], fname
        spots = configs[fname]
        return img, spots, fname

    base_bg, spots, bg_name = get_current_scene_data(current_bg_idx)
    
    print("\n--- COMENZI ---")
    print(" [SPACE] : Genereaza scenariu nou (Simulare + Inferenta)")
    print(" [B]     : Schimba parcarea (Background)")
    print(" [Q]     : Iesire")
    
    # Generam prima scena automat
    current_scene = base_bg.copy()
    display_img = current_scene.copy()
    need_update = True

    while True:
        # --- A. GENERARE SCENARIU (Daca e cerut) ---
        if need_update:
            current_scene = base_bg.copy()
            period = random.randint(0, 2) # 0=Morn, 1=Noon, 2=Eve
            occupancy_rate = random.uniform(0.3, 0.8)
            
            for spot in spots:
                if random.random() < occupancy_rate:
                    poly = np.array(spot, dtype=np.int32)
                    rect = cv2.boundingRect(poly)
                    x, y, w, h = rect
                    
                    if loaded_cars:
                        car = random.choice(loaded_cars)
                        # Unghi intre primele doua puncte
                        angle = np.degrees(np.arctan2(spot[1][1]-spot[0][1], spot[1][0]-spot[0][0]))
                        
                        scale = random.uniform(0.85, 0.95)
                        tw, th = int(w * scale), int(h * scale)
                        ox, oy = x + (w-tw)//2, y + (h-th)//2
                        
                        rotated_car = rotate_image(car, angle)
                        current_scene = overlay_transparent(current_scene, rotated_car, ox, oy, tw, th)

            # Efecte lumina
            if period == 1: current_scene = cv2.convertScaleAbs(current_scene, alpha=1.1, beta=10)
            elif period == 2: 
                current_scene = cv2.convertScaleAbs(current_scene, alpha=0.7, beta=-15)
                blue_overlay = np.full(current_scene.shape, (40, 20, 10), dtype='uint8')
                current_scene = cv2.addWeighted(current_scene, 0.8, blue_overlay, 0.2, 0)

            # --- B. INFERENTA ---
            display_img = current_scene.copy()
            overlay_mask = current_scene.copy()
            free_count = 0
            
            for spot in spots:
                poly = np.array(spot, dtype=np.int32)
                rect = cv2.boundingRect(poly)
                x, y, w, h = rect
                
                crop = current_scene[y:y+h, x:x+w]
                if crop.size == 0: continue

                # Preprocesare
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, predicted = torch.max(probs, 1)
                    label = predicted.item() # 0=Liber, 1=Ocupat (alfabetic)
                
                if label == 1: # Ocupat
                    cv2.fillPoly(overlay_mask, [poly], (0, 0, 255))
                else: # Liber
                    free_count += 1
                    cv2.polylines(display_img, [poly], True, (0, 255, 0), 2)

            cv2.addWeighted(overlay_mask, 0.35, display_img, 0.65, 0, display_img)
            
            # Info UI
            cv2.rectangle(display_img, (10, 10), (380, 100), (0,0,0), -1)
            cv2.putText(display_img, f"Parcare: {bg_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            period_str = ["Dimineata", "Pranz", "Seara"][period]
            cv2.putText(display_img, f"Moment: {period_str}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(display_img, f"Locuri Libere: {free_count}/{len(spots)}", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            need_update = False

        cv2.imshow("SMARTPark Demo (AI Powered)", display_img)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        elif key == 32: need_update = True
        elif key == ord('b'):
            current_bg_idx = (current_bg_idx + 1) % len(bg_names)
            base_bg, spots, bg_name = get_current_scene_data(current_bg_idx)
            need_update = True

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
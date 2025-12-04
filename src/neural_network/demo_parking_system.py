import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import json
import os
import glob
import random
from PIL import Image

# --- CONFIGURARE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(BASE_DIR, 'data', 'assets')
BG_DIR = os.path.join(ASSETS_DIR, 'backgrounds')
CAR_DIR = os.path.join(ASSETS_DIR, 'cars')
CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'synthetic_spots.json')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'parking_model.pth')
IMG_SIZE = 64

# --- ARHITECTURA CNN ---
class ParkingCNN(nn.Module):
    def __init__(self):
        super(ParkingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- FUNCȚII AJUTĂTOARE ---
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return result

def overlay_transparent(background, overlay, x, y, target_w, target_h):
    overlay_resized = cv2.resize(overlay, (target_w, target_h))
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay_resized.shape
    if x + w > bg_w or y + h > bg_h or x < 0 or y < 0: return background
    if overlay_resized.shape[2] < 4: return background
    b, g, r, a = cv2.split(overlay_resized)
    mask = a / 255.0
    roi = background[y:y+h, x:x+w]
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - mask) + overlay_resized[:, :, c] * mask
    background[y:y+h, x:x+w] = roi
    return background

def main():
    # 1. Încărcare Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rulez pe: {device}")
    
    model = ParkingCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        print("Modelul nu exista! Ruleaza train_cnn.py.")
        return
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Încărcare Resurse
    if not os.path.exists(CONFIG_FILE):
        print("Fisierul config nu exista! Ruleaza config_backgrounds.py.")
        return

    with open(CONFIG_FILE, 'r') as f:
        configs = json.load(f)
    
    # Lista cu numele fundalurilor disponibile
    available_bgs = list(configs.keys())
    if not available_bgs:
        print("Nu sunt fundaluri configurate in JSON.")
        return

    car_files = glob.glob(os.path.join(CAR_DIR, "*.png"))
    loaded_cars = [cv2.imread(c, cv2.IMREAD_UNCHANGED) for c in car_files]

    current_bg_index = 0
    
    # Funcție internă pentru a încărca fundalul curent
    def load_current_scene(index):
        bg_name = available_bgs[index]
        bg_path = os.path.join(BG_DIR, bg_name)
        img = cv2.imread(bg_path)
        spots = configs[bg_name]
        return img, spots, bg_name

    original_bg, spots, bg_name = load_current_scene(current_bg_index)
    
    print("--- COMENZI ---")
    print(" [SPACE] : Generează o situație nouă (mașini aleatorii)")
    print(" [B]     : Schimbă fundalul/parcarea")
    print(" [Q]     : Ieșire")

    # Flag pentru a genera o scenă nouă doar la cerere
    need_new_scene = True
    scene_img = None
    period = 0

    while True:
        if need_new_scene:
            scene_img = original_bg.copy()
            period = random.randint(0, 2) 
            occupancy_rate = random.uniform(0.3, 0.9)

            for spot in spots:
                if random.random() < occupancy_rate:
                    poly = np.array(spot, dtype=np.int32)
                    rect = cv2.boundingRect(poly)
                    x, y, w, h = rect
                    car_img = random.choice(loaded_cars)
                    
                    dx = spot[1][0] - spot[0][0]
                    dy = spot[1][1] - spot[0][1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    rotated_car = rotate_image(car_img, angle)
                    scale_factor = random.uniform(0.85, 0.95)
                    target_w = int(w * scale_factor)
                    target_h = int(h * scale_factor)
                    offset_x = x + (w - target_w)//2
                    offset_y = y + (h - target_h)//2
                    
                    scene_img = overlay_transparent(scene_img, rotated_car, offset_x, offset_y, target_w, target_h)

            # Efect lumina
            if period == 0: scene_img = cv2.convertScaleAbs(scene_img, alpha=1.0, beta=10)
            elif period == 1: scene_img = cv2.convertScaleAbs(scene_img, alpha=1.2, beta=20)
            elif period == 2: scene_img = cv2.convertScaleAbs(scene_img, alpha=0.7, beta=-10)
            
            need_new_scene = False # Nu mai generam pana nu se apasa SPACE sau B

        # --- CLASIFICARE SI VIZUALIZARE (ruleaza continuu) ---
        display_img = scene_img.copy()
        overlay = scene_img.copy()
        free_spots_count = 0
        
        for spot in spots:
            poly = np.array(spot, dtype=np.int32)
            rect = cv2.boundingRect(poly)
            x, y, w, h = rect
            crop = scene_img[y:y+h, x:x+w]
            
            if crop.size == 0: continue

            # Clasificare
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                label = predicted.item() 
            
            if label == 1: # Ocupat
                color = (0, 0, 255)
                cv2.fillPoly(overlay, [poly], color)
            else: # Liber
                color = (0, 255, 0)
                free_spots_count += 1
            
            cv2.polylines(display_img, [poly], True, color, 2)

        cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)

        # UI Text
        info_text = f"Locuri Libere: {free_spots_count} / {len(spots)}"
        cv2.rectangle(display_img, (20, 20), (550, 110), (0, 0, 0), -1)
        cv2.putText(display_img, info_text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        period_text = ["Dimineata", "Pranz", "Seara"][period]
        cv2.putText(display_img, f"Moment: {period_text}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Afisam numele parcarii curente
        cv2.putText(display_img, f"Parcare: {bg_name}", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("SMARTPark - Demo Live", display_img)
        
        key = cv2.waitKey(1) # Asteptam 1ms (non-blocking)

        if key == ord('q'):
            break
        elif key == 32: # SPACE key
            print("Generare scenariu nou...")
            need_new_scene = True
        elif key == ord('b'): # 'b' key pentru Background
            current_bg_index = (current_bg_index + 1) % len(available_bgs)
            original_bg, spots, bg_name = load_current_scene(current_bg_index)
            print(f"Schimbare parcare: {bg_name}")
            need_new_scene = True # Generam automat masini cand schimbam parcarea

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
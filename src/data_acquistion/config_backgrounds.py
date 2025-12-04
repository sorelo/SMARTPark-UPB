import cv2
import json
import os
import glob

# Căile relative
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BG_DIR = os.path.join(BASE_DIR, 'data', 'assets', 'backgrounds')
CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'synthetic_spots.json')

current_points = []
all_configs = {}  # Va stoca { "bg1.jpg": [[p1,p2,p3,p4], ...], "bg2.jpg": ... }

def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 4:
            current_points.append((x, y))

def main():
    global current_points, all_configs

    # Încărcăm config existent dacă e cazul
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            all_configs = json.load(f)

    # Găsim imaginile de fundal
    bg_files = glob.glob(os.path.join(BG_DIR, "*.jpg")) + glob.glob(os.path.join(BG_DIR, "*.png"))
    
    if not bg_files:
        print(f"Nu am găsit imagini în {BG_DIR}. Adaugă layout-urile acolo!")
        return

    print(f"Am găsit {len(bg_files)} fundaluri.")
    
    for bg_path in bg_files:
        filename = os.path.basename(bg_path)
        print(f"--> Configurare: {filename}")
        
        img = cv2.imread(bg_path)
        if img is None: continue

        # Dacă imaginea e prea mare, o redimensionăm doar pentru vizualizare, 
        # dar păstrăm coordonatele scalate? Nu, mai simplu lucrăm la rezoluția originală.
        # Dacă e imensă, folosește scroll sau zoom. Aici presupunem rezoluții decente.
        
        window_name = f"Configurare: {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, mouse_callback)

        # Dacă există deja locuri salvate pentru acest fundal, le încărcăm
        spots = all_configs.get(filename, [])

        while True:
            display = img.copy()
            
            # Desenăm locurile existente
            for spot in spots:
                for i in range(4):
                    cv2.line(display, tuple(spot[i]), tuple(spot[(i+1)%4]), (0, 255, 0), 2)

            # Desenăm punctele curente
            for pt in current_points:
                cv2.circle(display, pt, 5, (0, 0, 255), -1)
            
            if len(current_points) == 4:
                # Aratăm poligonul roșu înainte de salvare
                for i in range(4):
                    cv2.line(display, current_points[i], current_points[(i+1)%4], (0, 0, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'): # Save spot
                if len(current_points) == 4:
                    spots.append(current_points)
                    current_points = []
                    print(f"Loc adăugat. Total: {len(spots)}")
            elif key == ord('c'): # Clear current
                current_points = []
            elif key == ord('n'): # Next image
                all_configs[filename] = spots
                break # Ieșim din while, trece la următorul for
            elif key == ord('q'): # Quit all
                all_configs[filename] = spots
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(all_configs, f, indent=4)
                print("Configurare salvată și ieșire.")
                return

        cv2.destroyWindow(window_name)

    # Salvare finală
    with open(CONFIG_FILE, 'w') as f:
        json.dump(all_configs, f, indent=4)
    print("Configurare completă salvată în config/synthetic_spots.json")

if __name__ == "__main__":
    main()
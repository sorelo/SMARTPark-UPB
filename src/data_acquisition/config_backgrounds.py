import cv2
import json
import os
import glob
import numpy as np

# Configurare căi relative
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BG_DIR = os.path.join(BASE_DIR, 'data', 'assets', 'backgrounds')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'parking_spots.json')

# Variabile globale
current_points = []
all_configs = {}
ghost_spot = None # Locul care este în curs de clonare/mutare
drag_start = None
active_spots = [] # Lista locală de locuri pentru imaginea curentă

def mouse_callback(event, x, y, flags, param):
    global current_points, ghost_spot, drag_start, active_spots
    
    # --- CLICK STÂNGA: Adăugare puncte sau mutare clonă ---
    if event == cv2.EVENT_LBUTTONDOWN:
        if ghost_spot is not None:
            drag_start = (x, y)
        elif len(current_points) < 4:
            current_points.append([int(x), int(y)])
            print(f"Punct manual: ({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        if ghost_spot is not None and drag_start is not None:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            ghost_spot = [[p[0] + dx, p[1] + dy] for p in ghost_spot]
            drag_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drag_start = None

    # --- CLICK DREAPTA: Ștergere loc de sub cursor ---
    elif event == cv2.EVENT_RBUTTONDOWN:
        index_to_delete = -1
        for i, spot in enumerate(active_spots):
            # Verificăm dacă punctul (x,y) este în interiorul poligonului
            poly = np.array(spot, dtype=np.int32)
            dist = cv2.pointPolygonTest(poly, (x, y), False)
            if dist >= 0: # 0 sau pozitiv înseamnă în interior sau pe margine
                index_to_delete = i
                break
        
        if index_to_delete != -1:
            active_spots.pop(index_to_delete)
            print(f"Locul {index_to_delete + 1} a fost șters.")

def main():
    global current_points, all_configs, ghost_spot, active_spots
    
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            all_configs = json.load(f)

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(BG_DIR, ext)))

    if not image_paths:
        print("Eroare: Nu am găsit imagini în folderul data/assets/backgrounds!")
        return

    print("\n" + "="*60)
    print("INSTRUCȚIUNI:")
    print(" --- ADAUGARE ---")
    print(" 1. Click Stânga (4x): Trasează loc nou manual -> 'S' pt salvare.")
    print(" 2. 'D': Clonează ultimul loc salvat -> mută cu mouse-ul -> 'S'.")
    print(" --- ȘTERGERE & CORECȚIE ---")
    print(" 1. Click Dreapta: Șterge locul de sub cursor (cel verde).")
    print(" 2. 'Z': Undo (Șterge ultimul loc salvat).")
    print(" 3. 'C': Anulează selecția curentă sau clona albastră.")
    print("-" * 60)
    print(" N - Următoarea imagine | Q - Salvează tot și Ieși")
    print("="*60 + "\n")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue

        active_spots = all_configs.get(filename, [])
        cv2.namedWindow("Configurator SMARTPark", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Configurator SMARTPark", mouse_callback)

        while True:
            display = img.copy()
            
            # 1. Desenăm locurile deja salvate
            for s in active_spots:
                pts = np.array(s, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)

            # 2. Desenăm selecția manuală în curs\
            for pt in current_points:
                cv2.circle(display, tuple(pt), 4, (0, 0, 255), -1)
            
            if len(current_points) > 0 and len(current_points) < 4 and ghost_spot is None:
                cv2.putText(display, f"Manual: {len(current_points)}/4 puncte", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if len(current_points) == 4:
                pts = np.array(current_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (0, 0, 255), 2)
                cv2.putText(display, "Apasă 'S' pt salvare", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 3. Desenăm Clona
            if ghost_spot is not None:
                pts = np.array(ghost_spot, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (255, 255, 0), 2)
                cv2.putText(display, "Mod Clonare: Muta si apasa 'S' (sau 'C' pt renuntare)", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Statistici și Info
            cv2.putText(display, f"Imagine: {filename} | Locuri: {len(active_spots)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Click Dreapta pe loc pt stergerere | 'Z' pt Undo", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Configurator SMARTPark", display)
            key = cv2.waitKey(10) & 0xFF

            # S - Salvează
            if key == ord('s') or key == ord('S'):
                if ghost_spot is not None:
                    active_spots.append(list(ghost_spot))
                    print(f"Loc clonat salvat. Total: {len(active_spots)}")
                elif len(current_points) == 4:
                    active_spots.append(list(current_points))
                    current_points = []
                    print(f"Loc manual salvat. Total: {len(active_spots)}")

            # D - Duplicate
            elif key == ord('d') or key == ord('D'):
                if active_spots:
                    ghost_spot = [list(p) for p in active_spots[-1]]
                    current_points = []
                    # Mic offset
                    ghost_spot = [[p[0] + 10, p[1] + 10] for p in ghost_spot]
                else:
                    print("Eroare: Nu exista locuri de clonat.")

            # Z - Undo
            elif key == ord('z') or key == ord('Z'):
                if active_spots:
                    active_spots.pop()
                    print("Ultimul loc a fost eliminat (Undo).")

            # C - Clear / Cancel
            elif key == ord('c') or key == ord('C'):
                current_points = []
                ghost_spot = None
                print("Selecție curentă anulată.")

            # Sageti pt mutare fina
            if ghost_spot is not None:
                if key == 82 or key == 0:   ghost_spot = [[p[0], p[1]-1] for p in ghost_spot]
                elif key == 84 or key == 1: ghost_spot = [[p[0], p[1]+1] for p in ghost_spot]
                elif key == 81 or key == 2: ghost_spot = [[p[0]-1, p[1]] for p in ghost_spot]
                elif key == 83 or key == 3: ghost_spot = [[p[0]+1, p[1]] for p in ghost_spot]

            # N - Next Image
            elif key == ord('n') or key == ord('N'):
                all_configs[filename] = active_spots
                ghost_spot = None
                current_points = []
                break

            # Q - Save and Quit
            elif key == ord('q') or key == ord('Q'):
                all_configs[filename] = active_spots
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_configs, f, indent=4)
                print(f"Configurație salvată în {CONFIG_FILE}")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
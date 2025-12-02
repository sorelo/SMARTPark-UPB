import cv2
import json

CAMERA_INDEX = 0 

parking_spots = []
current_points = []

def mouse_callback(event, x, y, flags, param):
    global current_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 4:
            current_points.append((x, y))
            print(f"Punct adaugat: {x}, {y}")

def main():
    global current_points, parking_spots

    print(f"Incercam sa deschidem camera cu indexul {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Eroare: Nu s-a putut deschide camera {CAMERA_INDEX}.")
        print("Incearca sa schimbi CAMERA_INDEX in cod la 0 sau 2.")
        return

    print("--- INSTRUCTIUNI ---")
    print("1. Aranjeaza camera si macheta EXACT cum vor ramane.")
    print("2. Asteapta sa apara imaginea video.")
    print("3. Apasa tasta 'f' (Freeze) pentru a bloca imaginea si a incepe desenarea.")
    print("--------------------")

    frozen_frame = None
    drawing_mode = False

    cv2.namedWindow("Configurare Parcare")
    cv2.setMouseCallback("Configurare Parcare", mouse_callback)

    while True:
        if not drawing_mode:
            ret, frame = cap.read()
            if not ret: 
                print("Nu s-a putut citi cadrul. Verifica conexiunea USB.")
                break
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Cam {CAMERA_INDEX}: Apasa 'f' pt Freeze", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display_frame = frozen_frame.copy()
            
            for spot in parking_spots:
                for i in range(4):
                    pt1 = spot[i]
                    pt2 = spot[(i + 1) % 4]
                    cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)

            for pt in current_points:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            
            if len(current_points) > 1:
                for i in range(len(current_points) - 1):
                    cv2.line(display_frame, current_points[i], current_points[i+1], (0, 0, 255), 2)
            
            if len(current_points) == 4:
                cv2.line(display_frame, current_points[3], current_points[0], (0, 0, 255), 2)
                cv2.putText(display_frame, "Apasa 's' pt Salvare sau 'c' pt Anulare", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Configurare Parcare", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f') and not drawing_mode:
            print("Imagine inghetata. Deseneaza cele 4 colturi ale fiecarui loc (sens orar).")
            frozen_frame = frame
            drawing_mode = True
        elif key == ord('s'):
            if len(current_points) == 4:
                parking_spots.append(current_points)
                print(f"Locul {len(parking_spots)} salvat!")
                current_points = []
            else:
                print("Trebuie sa selectezi exact 4 colturi!")
        elif key == ord('c'):
            current_points = []
            print("Selectie anulata.")
        elif key == ord('w'):
            if len(parking_spots) > 0:
                with open('parking_config.json', 'w') as f:
                    json.dump(parking_spots, f)
                print(f"SALVAT! {len(parking_spots)} locuri salvate in 'parking_config.json'.")
                break
            else:
                print("Nu ai definit niciun loc.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
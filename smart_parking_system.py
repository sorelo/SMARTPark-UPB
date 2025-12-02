import cv2
import json
import numpy as np
from ultralytics import YOLO
import torch

# --- CONFIGURARE ---
CAMERA_INDEX = 0 
MIN_AREA = 2500      # Eliminam obiectele mai mici de atat
MIN_CONFIDENCE = 0.05 # 5% incredere minima
MAX_ASPECT_RATIO = 3.5 # Obiectele foarte lungi (benzi) sunt ignorate

def main():
    try:
        with open('parking_config.json', 'r') as f:
            parking_spots = json.load(f)
    except FileNotFoundError:
        print("Eroare: Nu s-a gasit 'parking_config.json'.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Se ruleaza pe: {device.upper()}")
    
    model = YOLO('yolo11m.pt')

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Eroare camera {CAMERA_INDEX}.")
        return

    print("Sistem pornit. Apasa 'q' pentru iesire.")
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    while True:
        ret, frame = cap.read()
        if not ret: break

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        frame_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        results = model(frame_enhanced, stream=True, device=device, conf=0.01, iou=0.5, verbose=False)

        occupied_spot_indices = set()

        valid_detections = []

        for r in results:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    if width == 0 or height == 0: continue
                    aspect_ratio = max(width, height) / min(width, height)

                    if area < MIN_AREA: continue
                    if aspect_ratio > MAX_ASPECT_RATIO: continue
                    if conf < MIN_CONFIDENCE: continue

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    is_inside_any_spot = False
                    
                    for i, spot in enumerate(parking_spots):
                        spot_pts = np.array(spot, np.int32).reshape((-1, 1, 2))
                        if cv2.pointPolygonTest(spot_pts, (center_x, center_y), False) >= 0:
                            is_inside_any_spot = True
                            occupied_spot_indices.add(i)
                            break

                    if is_inside_any_spot:
                        valid_detections.append((x1, y1, x2, y2, center_x, center_y, conf))

        overlay = frame.copy()

        for i, spot in enumerate(parking_spots):
            spot_pts = np.array(spot, np.int32).reshape((-1, 1, 2))
            
            if i in occupied_spot_indices:
                color = (0, 0, 255) # Rosu (Ocupat)
                # Umplem locul
                cv2.fillPoly(overlay, [spot_pts], color)
            else:
                color = (0, 255, 0) # Verde (Liber)
            
            cv2.polylines(frame, [spot_pts], True, color, 2)

        for (x1, y1, x2, y2, cx, cy, conf) in valid_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            label = f"Vehicul [{conf:.2f}]"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        free_spots = len(parking_spots) - len(occupied_spot_indices)
        total_spots = len(parking_spots)
        
        cv2.rectangle(frame, (20, 20), (450, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Locuri Libere: {free_spots} / {total_spots}", 
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("SmartPark UPB", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
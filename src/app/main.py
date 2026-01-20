import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
import json
import random
import pandas as pd
import time
from datetime import datetime
from PIL import Image

# --- CONFIGURARE CAI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.neural_network.model import ParkingCNN

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'optimized_model.pth')
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'parking_spots.json')
BG_DIR = os.path.join(BASE_DIR, 'data', 'assets', 'backgrounds')
CAR_DIR = os.path.join(BASE_DIR, 'data', 'assets', 'cars')
LOG_FILE = os.path.join(BASE_DIR, 'results', 'real_time_log.csv')

# --- MAPPING SI RELATII PROXIMITATE ---
ZONE_MAP = {
    "bg1.jpg": "Robotica",
    "bg2.jpg": "Automatica",
    "bg3.jpg": "Rectorat",
    "bg4.jpg": "Transporturi"
}

PROXIMITY_MAP = {
    "Rectorat": "Automatica",
    "Automatica": "Rectorat",
    "Robotica": "Transporturi",
    "Transporturi": "Robotica"
}

st.set_page_config(page_title="SMARTPark UPB - Dashboard", page_icon="P", layout="wide")

# --- STILIZARE VIZUALA AVANSATA (TEMA INTUNECATA) ---
st.markdown("""
    <style>
    /* Fundal general si font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
    
    .stApp {
        background-color: #0f172a;
        font-family: 'Segoe UI', sans-serif;
        color: #f1f5f9;
    }

    /* Header */
    h1 {
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-top: -20px !important;
    }
    
    .status-badge {
        background-color: #1e293b;
        color: #deff9a;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid #334155;
        font-weight: bold;
    }

    /* Carduri de Metrici in Statistici */
    [data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #deff9a !important;
        font-weight: 700 !important;
    }

    /* Caseta de Recomandare */
    .recommendation-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 8px solid #deff9a;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    .recommendation-title {
        color: #deff9a;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    
    .recommendation-text {
        color: #cbd5e1;
        font-size: 1.1rem;
        line-height: 1.5;
    }

    /* Tab-uri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #1e293b;
        border-radius: 8px 8px 0 0;
        padding: 0 24px;
        color: #94a3b8;
        border: 1px solid #334155;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #deff9a !important;
        color: #0f172a !important;
    }

    /* Imagini harti */
    .stImage img {
        border-radius: 10px;
        border: 1px solid #334155;
    }

    /* Buton principal */
    .stButton button {
        background-color: #deff9a !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 700 !important;
        height: 3.5rem !important;
        border-radius: 10px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: 0.3s;
    }
    
    .stButton button:hover {
        background-color: #e2ffae !important;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGICA BACKEND (Sincronizata) ---
def get_time_period(hour):
    if 7 <= hour <= 11: return "Dimineata"
    if 12 <= hour <= 16: return "Pranz"
    return "Seara"

def log_ai_results(results, hour):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = []
    for res in results:
        new_data.append({
            "Timestamp": timestamp,
            "Ora_Selectata": hour,
            "Zona": res["nume"],
            "Locuri_Libere": res["libere"],
            "Grad_Ocupare": round((res["total"] - res["libere"]) / res["total"] * 100, 2)
        })
    df_new = pd.DataFrame(new_data)
    
    should_overwrite = False
    if os.path.exists(LOG_FILE):
        try:
            temp_df = pd.read_csv(LOG_FILE, nrows=0)
            if len(temp_df.columns) != len(df_new.columns): should_overwrite = True
        except: should_overwrite = True

    if not os.path.exists(LOG_FILE) or should_overwrite:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        df_new.to_csv(LOG_FILE, index=False)
    else:
        df_new.to_csv(LOG_FILE, mode='a', header=False, index=False)

@st.cache_resource
def get_ai_brain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParkingCNN().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    return model, device

def apply_ai_inference(image_crop, model, device):
    img = cv2.resize(image_crop, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def simulate_environment(image, period):
    if period == "Pranz": return cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    if period == "Seara":
        image = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)
        overlay = np.full(image.shape, (40, 20, 10), dtype='uint8')
        return cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
    return image

def analyze_all_zones(configs, model, device, hour):
    period = get_time_period(hour)
    zone_results = []
    car_files = [cv2.imread(c, -1) for c in [os.path.join(CAR_DIR, f) for f in os.listdir(CAR_DIR) if f.endswith('.png')]]
    
    for bg_file, spots in configs.items():
        img_path = os.path.join(BG_DIR, bg_file)
        base_img = cv2.imread(img_path)
        if base_img is None: continue
        
        temp_scene = base_img.copy()
        occ_chance = random.uniform(0.1, 0.6) if period != "Pranz" else random.uniform(0.5, 0.95)
        
        for spot in spots:
            if random.random() < occ_chance:
                poly = np.array(spot, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(poly)
                if car_files:
                    car = cv2.resize(random.choice(car_files), (w, h))
                    alpha = car[:,:,3]/255.0
                    for c in range(3): temp_scene[y:y+h, x:x+w, c] = temp_scene[y:y+h, x:x+w, c]*(1-alpha) + car[:,:,c]*alpha

        final_scene = simulate_environment(temp_scene, period)
        free_count = 0
        draw_img = final_scene.copy()
        overlay = final_scene.copy()
        
        for spot in spots:
            poly = np.array(spot, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(poly)
            crop = final_scene[y:y+h, x:x+w]
            if crop.size == 0: continue
            is_occupied = apply_ai_inference(crop, model, device)
            if is_occupied == 0:
                free_count += 1
                cv2.polylines(draw_img, [poly], True, (0, 255, 0), 2)
            else:
                cv2.fillPoly(overlay, [poly], (0, 0, 255))
        
        cv2.addWeighted(overlay, 0.4, draw_img, 0.6, 0, draw_img)
        friendly_name = ZONE_MAP.get(bg_file, bg_file)
        zone_results.append({
            "nume": friendly_name,
            "total": len(spots),
            "libere": free_count,
            "imagine": cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        })
    
    log_ai_results(zone_results, hour)
    return zone_results

# --- UI MAIN ---
def main():
    st.markdown("<h1>SMARTPark UPB <span class='status-badge'>Sistem Activ</span></h1>", unsafe_allow_html=True)
    
    model, device = get_ai_brain()
    if not os.path.exists(CONFIG_PATH):
        st.error("Configuratia lipseste din folderul config.")
        return
    with open(CONFIG_PATH, 'r') as f: configs = json.load(f)

    tab1, tab2 = st.tabs(["Asistent Planificare", "Analiza Grad Ocupare"])

    # --- TAB 1: ASISTENT ---
    with tab1:
        c1, c2 = st.columns([1, 2.5], gap="large")
        with c1:
            st.markdown("### Optiuni Calatorie")
            target_hour = st.slider("Ora sosirii in campus", 7, 21, 9)
            
            preferred_dest = st.selectbox(
                "Destinatia vizata",
                options=list(ZONE_MAP.values())
            )
            
            st.markdown(f"<p style='color: #94a3b8;'>Moment detectat: <b>{get_time_period(target_hour)}</b></p>", unsafe_allow_html=True)
            search_btn = st.button("Verifica Disponibilitatea", use_container_width=True)

        with c2:
            if search_btn:
                with st.spinner("Motorul AI analizeaza fluxul video..."):
                    results = analyze_all_zones(configs, model, device, target_hour)
                    
                pref_data = next((r for r in results if r["nume"] == preferred_dest), None)
                buddy_name = PROXIMITY_MAP.get(preferred_dest)
                buddy_data = next((r for r in results if r["nume"] == buddy_name), None) if buddy_name else None
                
                title_rec, msg_rec = "", ""
                
                if pref_data and pref_data["libere"] > 0:
                    title_rec = f"Loc Gasit la {preferred_dest}"
                    msg_rec = f"Exista {pref_data['libere']} locuri libere identificate. Va puteti indrepta direct catre parcare."
                elif buddy_data and buddy_data["libere"] > 0:
                    title_rec = f"Parcarea {preferred_dest} este Plina"
                    msg_rec = f"Va recomandam parcarea {buddy_name}. Este cea mai apropiata varianta si are {buddy_data['libere']} locuri libere."
                else:
                    best_alt = max(results, key=lambda x: x['libere'])
                    title_rec = "Aglomeratie Ridicata"
                    msg_rec = f"Zonele din apropiere sunt ocupate. Cea mai buna alternativa acum este {best_alt['nume']}."

                st.markdown(f"""
                    <div class="recommendation-container">
                        <div class="recommendation-title">{title_rec}</div>
                        <div class="recommendation-text">{msg_rec}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Afisare imagini in grid
                grid_cols = st.columns(2)
                for i, r in enumerate(results):
                    with grid_cols[i % 2]:
                        st.image(r['imagine'], caption=f"{r['nume']} | {r['libere']} Libere", use_container_width=True)
            else:
                st.markdown("""
                    <div style='text-align: center; padding: 100px 20px; color: #64748b;'>
                        <h3>Sistem in Asteptare</h3>
                        <p>Configurati ora sosirii si destinatia pentru a rula analiza neurala.</p>
                    </div>
                """, unsafe_allow_html=True)

    # --- TAB 2: STATISTICI ---
    with tab2:
        if os.path.exists(LOG_FILE):
            try:
                df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
                st.markdown("### Monitorizare Date Istorice")
                
                sel_hour = st.select_slider("Filtreaza media de ocupare pe ora", options=range(7, 22), value=12)
                df_h = df[df['Ora_Selectata'] == sel_hour]
                
                if not df_h.empty:
                    stats = df_h.groupby('Zona')['Grad_Ocupare'].mean().reset_index()
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Ocupare Medie", f"{stats['Grad_Ocupare'].mean():.1f}%")
                    m2.metric("Total Probe AI", len(df_h))
                    m3.metric("Ora Analizata", f"{sel_hour}:00")
                    
                    st.divider()
                    
                    col_l, col_r = st.columns([1.5, 1])
                    with col_l:
                        st.markdown(f"#### Comparatie Zone la ora {sel_hour}:00")
                        st.bar_chart(stats.set_index('Zona'))
                    with col_r:
                        st.markdown("#### Detalii Procentuale")
                        st.table(stats.set_index('Zona'))
                    
                    st.divider()
                    st.markdown("#### Evolutia Gradului de Ocupare (Total)")
                    line_chart_data = df.pivot_table(index='Ora_Selectata', columns='Zona', values='Grad Ocupare', aggfunc='mean')
                    st.line_chart(line_chart_data)
                else:
                    st.warning(f"Nu exista date inregistrate pentru ora {sel_hour}:00. Efectuati cautari pentru a popula baza de date.")
            
            except Exception:
                st.error("Eroare la procesarea fisierului de log. Incercati sa resetati datele.")
                if st.button("Resetare Log"):
                    os.remove(LOG_FILE)
                    st.rerun()
        else:
            st.warning("Inca nu au fost colectate date. Rulati asistentul de planificare.")

if __name__ == "__main__":
    main()

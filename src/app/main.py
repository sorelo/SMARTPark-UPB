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

# --- path-uri ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.neural_network.model import ParkingCNN

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.pth')
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'parking_spots.json')
BG_DIR = os.path.join(BASE_DIR, 'data', 'assets', 'backgrounds')
CAR_DIR = os.path.join(BASE_DIR, 'data', 'assets', 'cars')
LOG_FILE = os.path.join(BASE_DIR, 'results', 'real_time_log.csv')

# --- relatii ---
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

# --- stilizare css ---
st.markdown("""
    <style>
    /* Reset si Fundal General */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
        color: #f8f9fa;
    }

    /* Header si Titlu */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -1px;
        color: #ffffff !important;
        margin-bottom: 0px !important;
    }
    
    .ai-badge {
        background: rgba(222, 255, 154, 0.1);
        color: #DEFF9A;
        border: 1px solid #DEFF9A;
        padding: 4px 14px;
        border-radius: 30px;
        font-size: 0.4em;
        vertical-align: middle;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-left: 15px;
    }

    /* Carduri de Metrici */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        padding: 25px !important;
        border-radius: 16px !important;
        transition: transform 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(222, 255, 154, 0.4) !important;
    }

    [data-testid="stMetricValue"] {
        color: #DEFF9A !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }

    /* Caseta Recomandare */
    .recommendation-box {
        background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        padding: 30px;
        border-radius: 20px;
        border-left: 6px solid #DEFF9A;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .recommendation-box h3 {
        color: #DEFF9A !important;
        margin-bottom: 10px !important;
        text-transform: uppercase;
        font-size: 1.1rem !important;
        letter-spacing: 1px;
    }

    /* Tab-uri Personalizate */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        margin-bottom: 30px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: #94a3b8;
        padding: 0 30px;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #DEFF9A !important;
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Imagini harti */
    .stImage img {
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Sidebar si Widget-uri */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    .stSlider [data-baseweb="slider"] {
        padding-top: 25px;
    }

    .stButton button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        height: 3rem !important;
        border: none !important;
        background: #DEFF9A !important;
        color: #0f172a !important;
        box-shadow: 0 4px 15px rgba(222, 255, 154, 0.2) !important;
    }
    
    .stButton button:hover {
        background: #eaffbc !important;
        box-shadow: 0 6px 20px rgba(222, 255, 154, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- logica functii ---
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
            "nume": friendly_name, "total": len(spots), "libere": free_count,
            "imagine": cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        })
    log_ai_results(zone_results, hour)
    return zone_results

# --- main ui ---
def main():
    st.markdown("<h1>SMARTPark UPB</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 40px;'>Sistem de monitorizare campus inteligent bazat pe inteligenta artificiala.</p>", unsafe_allow_html=True)
    
    model, device = get_ai_brain()
    if not os.path.exists(CONFIG_PATH):
        st.error("Eroare: Fisierul de configurare lipseste.")
        return
    with open(CONFIG_PATH, 'r') as f: configs = json.load(f)

    tab1, tab2 = st.tabs(["Planificare", "Statistici"])

    with tab1:
        c1, c2 = st.columns([1, 2.2], gap="large")
        with c1:
            st.markdown("### Configuratie Sosire")
            target_hour = st.slider("Ora estimata", 7, 21, 9)
            preferred_dest = st.selectbox("Destinatie campus", options=list(ZONE_MAP.values()))
            st.markdown(f"<div style='padding:10px; border-radius:10px; background:rgba(255,255,255,0.05); text-align:center;'>Status: <b>{get_time_period(target_hour)}</b></div>", unsafe_allow_html=True)
            search_btn = st.button("Analizeaza Disponibilitate", use_container_width=True)

        with c2:
            if search_btn:
                with st.spinner("Motorul AI analizeaza pixelii zonelor de parcare..."):
                    results = analyze_all_zones(configs, model, device, target_hour)
                
                pref_data = next((r for r in results if r["nume"] == preferred_dest), None)
                buddy_name = PROXIMITY_MAP.get(preferred_dest)
                buddy_data = next((r for r in results if r["nume"] == buddy_name), None) if buddy_name else None
                
                rec_title, rec_msg = "", ""
                if pref_data and pref_data["libere"] > 0:
                    rec_title = f"PARCARE DISPONIBILA: {preferred_dest}"
                    rec_msg = f"Am identificat {pref_data['libere']} locuri libere la destinatia dumneavoastra."
                elif buddy_data and buddy_data["libere"] > 0:
                    rec_title = f"RECOMANDARE ALTERNATIVA: {buddy_name}"
                    rec_msg = f"Zona {preferred_dest} este aglomerata. {buddy_name} este cea mai apropiata parcare libera ({buddy_data['libere']} locuri)."
                else:
                    best_alt = max(results, key=lambda x: x['libere'])
                    rec_title = "CAPACITATE LIMITATA IN ZONA"
                    rec_msg = f"Zonele apropiate sunt ocupate. Va sugeram parcarea {best_alt['nume']}."

                st.markdown(f"""
                    <div class="recommendation-box">
                        <h3>{rec_title}</h3>
                        <p>{rec_msg}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(2)
                for i, r in enumerate(results):
                    with cols[i%2]:
                        st.image(r['imagine'], caption=f"{r['nume']} | {r['libere']} locuri", use_container_width=True)
            else:
                st.info("Configurati ora si destinatia pentru a rula analiza neurala.")

    with tab2:
        if os.path.exists(LOG_FILE):
            try:
                df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
                st.markdown("### Media Gradului de Ocupare")
                filter_hour = st.select_slider("Filtreaza istoricul pe ora", options=range(7, 22), value=12)
                df_filtered = df[df['Ora_Selectata'] == filter_hour]
                
                if not df_filtered.empty:
                    stats = df_filtered.groupby('Zona')['Grad_Ocupare'].mean().reset_index()
                    k1, k2 = st.columns([2, 1])
                    with k1:
                        st.bar_chart(stats.set_index('Zona'))
                    with k2:
                        for _, row in stats.iterrows():
                            st.metric(row['Zona'], f"{row['Grad_Ocupare']:.1f}%")
                    
                    st.divider()
                    st.markdown("### Tendinte Zilnice")
                    line_data = df.pivot_table(index='Ora_Selectata', columns='Zona', values='Grad_Ocupare', aggfunc='mean')
                    st.line_chart(line_data)
                else:
                    st.warning(f"Nu exista inregistrari AI pentru ora {filter_hour}:00 in baza de date.")
            except Exception as e:
                st.error("Structura bazei de date a fost modificata.")
                if st.button("Curata Date Vechi"):
                    os.remove(LOG_FILE); st.rerun()
        else:
            st.warning("Efectuati o cautare in tab-ul de Planificare pentru a genera statistici.")

if __name__ == "__main__":
    main()
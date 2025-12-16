# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ilie Marian-Ionuț  
**Link Repository GitHub:** https://github.com/sorelo/SMARTPark-UPB  
**Data:** 12.12.2025  

---

## Scopul Etapei 4

Această etapă corespunde punctului 5. Dezvoltarea arhitecturii aplicației software bazată pe RN.  
Am livrat un **SCHELET COMPLET** și **FUNCȚIONAL** al întregului Sistem cu Inteligență Artificială (SIA).

---

## Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Reducerea timpului de căutare a parcării în campusul UPB (actual > 10 min) | Monitorizare video și clasificare automată (Liber/Ocupat) cu timp de răspuns < 0.5 secunde | `src/neural_network/demo_parking_system.py` (Inference & UI) |
| Lipsa unui dataset diversificat pentru condiții variabile (lumină, unghi, aglomerare) | Generare Sintetică: Crearea automată a 3.600+ scenarii realiste prin augmentare digitală | `src/data_acquisition/generate_synthetic_data.py` |
| Adaptabilitate rapidă la reconfigurarea parcărilor existente | Sistem configurabil software prin definirea regiunilor de interes (ROI) pe imagini statice | `src/data_acquisition/config_backgrounds.py` |

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40%

**Contribuția originală la setul de date:**

- **Total observații finale:** 3.600 imagini (după Etapa 3 + Etapa 4)  
- **Observații originale:** 3.600 (100% generate sintetic)

**Tipul contribuției:**

- [ ] Date generate prin simulare fizică  
- [ ] Date achiziționate cu senzori proprii  
- [ ] Etichetare/adnotare manuală  
- [x] Date sintetice prin metode avansate  

**Descriere detaliată:**  
Deși elementele grafice brute (imagini satelit, sprites mașini) provin din surse externe, dataset-ul final este o creație originală rezultată din procesarea acestora printr-un pipeline software dezvoltat personal.

Contribuția inginerească constă în dezvoltarea unui **Motor de Date Sintetice** care:

- **Automatizează plasarea:** Calculează unghiul de rotație necesar pentru fiecare loc de parcare individual pe baza geometriei fundalului.
- **Simulează mediul:** Implementează algoritmi de procesare a imaginii (ajustare gamma, color mapping) pentru a crea variații de iluminare (dimineață, prânz, seară).
- **Etichetează automat:** Generează implicit etichetele (ground truth: 0/1) fără eroare umană, prin controlul programatic al procesului de generare.

**Locația codului:** `src/data_acquisition/generate_synthetic_data.py`  
**Locația datelor:** `data/processed/` (folderele `liber` și `ocupat`)

**Dovezi:**
- Scriptul de generare este funcțional și produce structura de directoare.
- Grafic distribuție: `docs/datasets/dataset_distribution.png`

---

### 3. Diagrama State Machine a Întregului Sistem

**Diagrama Vizuală:** Consultați fișierul `docs/state_machine.png`

**Descrierea fluxului:**

```
START → INITIALIZE_RESOURCES (Load Model, Configs, Assets)
→ WAIT_USER_INPUT (Idle Loop)
  ├─ [Key: SPACE] → GENERATE_SCENARIO (Simulare mașini + lumină)
  │                 → EXTRACT_ROIs (Decupare locuri)
  │                 → PREPROCESS (Resize 64x64, Norm)
  │                 → CNN_INFERENCE (Batch prediction)
  │                 → UPDATE_OVERLAY (Draw Red/Green)
  │                 → DISPLAY_RESULT
  │                 → WAIT_USER_INPUT
  │
  ├─ [Key: B] → CHANGE_BACKGROUND (Load next layout)
  │             → GENERATE_SCENARIO ...
  │
  └─ [Key: Q] → CLEANUP → STOP
```

---

### 4. Scheletul Complet al celor 3 Module

| **Modul** | **Tehnologie** | **Status Implementare** |
|----------|---------------|--------------------------|
| 1. Data Acquisition | Python (OpenCV, NumPy) | Complet Funcțional. |
| 2. Neural Network | Python (PyTorch) | Definit & Compilat. |
| 3. Web Service / UI | Python (OpenCV HighGUI) | Funcțional. |

---

## Structura Repository-ului la Finalul Etapei 4

```
SMARTPARK-UPB/
├── data/
├── src/
├── docs/
├── config/
├── README.md
├── README_Etapa3.md
├── README_Etapa4_Arhitectura_SIA.md
└── requirements.txt
```

---

## Checklist Final

- [x] Tabel Nevoie → Soluție → Modul
- [x] Contribuție 100% date originale
- [x] Diagramă State Machine
- [x] UI funcțional

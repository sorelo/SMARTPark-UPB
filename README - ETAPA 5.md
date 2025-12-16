# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ilie Marian-Ionuț  
**Link Repository GitHub:** https://github.com/sorelo/SMARTPark-UPB  
**Data predării:** 16.12.2025  

---

## Scopul Etapei 5

Această etapă corespunde punctului 6. Configurarea și antrenarea modelului RN.  
Obiectivul este antrenarea efectivă a modelului CNN definit în Etapa 4 pe dataset-ul sintetic generat, evaluarea performanței acestuia și integrarea modelului antrenat (.pth) în aplicația finală de monitorizare.

**Pornire:** Arhitectura completă din Etapa 4, cu dataset-ul de 3.600 imagini sintetice.

---

## PREREQUISITE – Verificare Etapa 4

- [x] State Machine definit în `docs/state_machine.png`.
- [x] Contribuție 100% date originale (generate sintetic) în `data/processed/`.
- [x] Modul 1 (Data Generation) funcțional – `generate_synthetic_data.py`.
- [x] Modul 2 (RN) cu arhitectură definită (`ParkingCNN`).
- [x] Modul 3 (UI) funcțional (`demo_parking_system.py`).

---

## Pregătire Date pentru Antrenare

Dataset-ul a fost generat și preprocesat în etapele anterioare. În această etapă, scriptul de antrenare preia datele gata împărțite.

### Structura curentă a datelor

- **Train (70%)**: 2.520 imagini (pentru ajustarea greutăților)
- **Validation (15%)**: 540 imagini (pentru monitorizarea epocilor)
- **Test (15%)**: 540 imagini (pentru calculul metricilor finale)

### Preprocesare aplicată (în `train_cnn.py`)

- Resize la 64x64 pixeli
- Normalizare: `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`
- Conversie la Tensor PyTorch

---

## Cerințe Nivel 1 & 2

### 1. Tabel Hiperparametri și Justificări

| Hiperparametru | Valoare Aleasă | Justificare |
|---------------|---------------|-------------|
| Learning Rate | 0.001 | Valoare standard pentru optimizatorul Adam. Asigură o convergență rapidă dar stabilă pentru arhitecturi CNN superficiale. |
| Batch Size | 32 | Avem N=3.600 samples. Oferă un echilibru bun între viteza de execuție și stabilitatea gradientului. |
| Number of Epochs | 10 | Dataset sintetic curat. Modelul converge rapid, fără risc major de overfitting. |
| Optimizer | Adam | Ajustează rata de învățare per parametru, evitând minime locale. |
| Loss Function | CrossEntropyLoss | Standard pentru probleme de clasificare. |
| Architecture | 3x Conv Layers | Imaginile 64x64 nu necesită rețele adânci. |

---

### 2. Rezultate Antrenare (Metrici)

În urma rulării `src/neural_network/train_cnn.py`:

- **Training Accuracy:** 98.5%
- **Validation Accuracy:** 97.2%
- **Test Accuracy:** 96.8% (≥ 65% cerință)
- **F1-score (macro):** 0.96 (≥ 0.60 cerință)

Modelul a fost salvat cu succes în:  
`data/parking_model.pth`

---

### 3. Integrare în UI

Aplicația `demo_parking_system.py` a fost actualizată pentru a încărca fișierul `parking_model.pth`.

Predicțiile *Liber/Ocupat* sunt realizate de rețeaua antrenată.

**Dovadă:** `docs/screenshots/inference_real.png`

---

## Analiză Erori în Context Industrial (Nivel 2)

### 1. Pe ce clase greșește cel mai mult modelul?

Confuzia principală apare în direcția **False Negative** (Predicție: Liber | Realitate: Ocupat).

### 2. Ce caracteristici ale datelor cauzează erori?

- Iluminare extremă (scenariul „Seara”)
- Mașini de culoare închisă
- Ocluzie parțială

### 3. Ce implicații are pentru aplicația industrială?

- **False Positive:** Impact acceptabil
- **False Negative:** Impact CRITIC

**Prioritate:** Minimizarea False Negatives.

### 4. Ce măsuri corective propuneți?

- Augmentare avansată (zgomot, low-light)
- Class Weights în funcția de loss
- Threshold dinamic (probabilitate > 0.3 → Ocupat)

---

## Structura Repository-ului la Finalul Etapei 5

```
SMARTPARK-UPB/
├── README.md
├── README_Etapa3.md
├── README_Etapa4_Arhitectura_SIA.md
├── README_Etapa5_Antrenare.md
├── docs/
│   ├── state_machine.png
│   ├── loss_curve.png
│   └── screenshots/
│       └── inference_real.png
├── data/
│   ├── assets/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── parking_model.pth
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   └── neural_network/
│       ├── train_cnn.py
│       ├── evaluate_cnn.py
│       └── demo_parking_system.py
├── config/
│   └── synthetic_spots.json
└── requirements.txt
```

---

## Instrucțiuni de Rulare și Verificare

### 1. Antrenare Model

```bash
python src/neural_network/train_cnn.py
```

### 2. Evaluare

```bash
python src/neural_network/train_cnn.py
```

### 3. Lansare Aplicație

```bash
python src/neural_network/demo_parking_system.py
```

---

## Checklist Final Etapa 5

- [x] Model antrenat de la zero
- [x] Hiperparametri justificați
- [x] Acuratețe > 65%
- [x] UI integrat cu inferență reală
- [x] Analiză erori realizată
- [x] Grafice de antrenare salvate

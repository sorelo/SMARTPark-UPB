# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ilie Marian-Ionut  
**Data:** 20.11.2025  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
SMARTPARK-UPB/
├── README.md
├── docs/
│   └── datasets/          # grafice distributie, descriere tehnica
├── data/
│   ├── raw/               # resurse: fundaluri parcare, imagini masini (png.)
│   ├── processed/         # date generate (crop-uri 64x64)
│   ├── train/             # set de instruire (organizat pe clase)
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # scripturi: split_dataset.py, visualize_stats.py
│   ├── data_acquisition/  # scripturi: config_backgrounds.py, generate_synthetic_data.py
│   └── neural_network/    # implementarea RN (train_cnn.py) 
├── config/                # synthetic_spots.json (coordonate locuri)
└── requirements.txt       # dependențe Python (torch, cv2, etc.)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Date Sintetice generate programatic. S-au utilizat 4 layout-uri de parcare (schițe/poze reale) și 4 asset-uri de autovehicule (fotografiate top-down și decupate).
* **Modul de achiziție:** ☑ Generare programatică (Script Python cu OpenCV).
* **Perioada / condițiile colectării:** Noiembrie 2025. S-au simulat algoritmic 3 condiții de iluminare: Dimineața (neutru), Prânz (contrast ridicat), Seara (luminozitate scăzută, tentă albastră).

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 3600 imagini generate.
* **Număr de caracteristici (features):** Input: Matrice de pixeli (64x64x3).
* **Tipuri de date:** ☑ Imagini (Numerice - Tensori).
* **Format fișiere:** ☑ JPG (Imagini), JSON (Metadate/Configurare).

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** |  **Tip**   |  **Unitate**  |                        **Descriere**                       | **Domeniu valori** |
|--------------------|------------|---------------|------------------------------------------------------------|--------------------|
| Pixel (R,G,B)      | Numeric    |  Intensitate  | Valoarea bruta a pixelului preluat de senzor               |   0–255 (uint8)    |
| Pixel Normalizat   | Numeric    |      -        | Valorea pixelului dupa standardizare (Mean=0.5, Std=0.5)   | -1.0-1.0 (float)   |
| Label(Clasa)       | Categorial |      -        | Eticheta locului de parcare (Target de predictie)          | 0: Liber, 1: Ocupa |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Distributia claselor:** Dataset-ul este perfect echilibrat prin constructie (algorimtul a generat un numar egal de instante pentru fiecare scenariu).
* **Variatia iluminarii:** 33% Dimineata, 33% Pranz, 33% Seara.
* **Distribuții pe caracteristici** (histograme)
* **Dimensiuni:** Toate imaginile sunt standardizate la 64x64 pixeli.

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă:** Nu exista (date sintetice).
* **Consistenta:** Toate imaginile au aceeasi rezolutie si adancime de culoare (3 canale).
* **Identificarea artefactelor:** Verificarea vizuala a masinilor generate la marginea locurilor de parcare (clipping).

### 3.3 Probleme identificate si solutii

* Problema: Riscul de overfitting pe formele specifice ale celor 4 masini folosite.
  * Solutie: S-a aplicat augmentare geometrica (rotatie aleatorie, scalare 85-95%) in momentul generarii.

* Problema: Similitudine mare intre cadrele succesive.
  * Solutie: split_dataset.py foloseste random.shuffle() inainte de impartire pentru a asigura diversitatea.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Filtrare:** Scriptul de generare elimina automat crop-urile care au dimensiuni nule sau invalide (care ies din cadru imaginii de fundal).

### 4.2 Transformarea caracteristicilor

* **Redimensionare:** Toate crop-urile sunt aduse la 64x64 pixeli.

* **Conversie Tensor:** Transformarea din matrice NumPy (H, W, C) în Tensor PyTorch (C, H, W).

* **Normalizare:** Aplicarea standardizării: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] pentru a centra datele și a ajuta convergența CNN-ului.

### 4.3 Structurarea seturilor de date

S-a utilizat scriptul split_dataset.py pentru a imparti datele din data/processed/ in:

* **Train(70%):** 2520 imagini (pentru atrenarea greutatilor).
* **Validation(15%):** 540 imagini (pentru monitorizarea epocilor si prevenirea overfitting).
* **Test(15%):** 540 imagini (pentru evalurea finala).

### 4.4 Salvarea rezultatelor preprocesării

* **Structura finala pe disc:**
  * data/train/liber & data/train/ocupat
  * data/validation/liber & data/validation/ocupat
  * data/test/liber & data/test/ocupat

---

##  5. Fișiere Generate în Această Etapă

* src/data_acquisition/generate_synthetic_data.py – motorul de generare date.
* src/preprocessing/split_dataset.py – utilitarul de împărțire train/validare/test.
* src/preprocessing/visualize_stats.py – generatorul de grafice.
* config/synthetic_spots.json – coordonatele ROI pentru generare.
* data/processed/ – repository-ul cu cele 3600 imagini brute.

---

##  6. Stare Etapă (de completat de student)

- ☑ Structură repository configurată
- ☑ Dataset analizat (EDA realizată)
- ☑ Date preprocesate
- ☑ Seturi train/val/test generate
- ☑ Documentație actualizată în README + `data/README.md`

---

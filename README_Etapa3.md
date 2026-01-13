# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date

**Disciplina:** Rețele Neuronale  
**Student:** Ilie Marian-Ionuț  
**Data:** 20.11.2025  

---

## 1. Structura Repository (Etapa 3)
*(Vezi structura de foldere generată pe disc)*

## 2. Descrierea Setului de Date
### 2.1 Sursa datelor
* **Origine:** Date Sintetice generate programatic.
* **Modul de achiziție:** ☑ Generare programatică (Script Python `generate_synthetic_data.py`).
* **Perioada:** Noiembrie 2025.

### 2.2 Caracteristici
* **Volum:** 3.600 imagini (64x64 pixeli).
* **Clase:** Echilibrat (50% Liber, 50% Ocupat).
* **Format:** JPG (Imagini), JSON (Configurare).

### 2.3 Descriere Features
| Caracteristică | Tip | Descriere | Domeniu |
|---|---|---|---|
| Pixel RGB | Numeric | Intensitate | 0-255 |
| Label | Categorial | Clasa (Target) | 0/1 |

## 3. Analiza Exploratorie (EDA)
* **Distribuție:** Perfect echilibrată prin construcție.
* **Variație:** 3 condiții de iluminare simulate (Dimineața, Prânz, Seara).
* **Calitate:** Fără valori lipsă (date sintetice).

## 4. Preprocesare
* **Resize:** 64x64 px.
* **Normalizare:** Standardizare (Mean=0.5, Std=0.5).
* **Split:** 70% Train, 15% Val, 15% Test.

## 5. Fișiere Generate
* `src/data_acquisition/generate_synthetic_data.py`
* `src/preprocessing/split_dataset.py`
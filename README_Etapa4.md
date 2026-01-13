# 📘 README – Etapa 4: Arhitectura Completă SIA

**Disciplina:** Rețele Neuronale  
**Student:** Ilie Marian-Ionuț  
**Data:** 12.12.2025  

---

## Scop
Livrarea unui schelet complet și funcțional al sistemului SIA.

## 1. Tabel Nevoie -> Soluție
| Nevoie | Soluție SIA | Modul |
|---|---|---|
| Timp căutare parcare ridicat | Monitorizare video real-time (<0.5s) | `src/neural_network/demo_parking_system.py` |
| Lipsa datelor variate | Generare Sintetică (3600+ scenarii) | `src/data_acquisition/generate_synthetic_data.py` |

## 2. Contribuția Originală (40%)
* **Total:** 3.600 imagini.
* **Originale:** 3.600 (**100%**).
* **Tip:** [x] Date sintetice prin metode avansate.
* **Descriere:** Motor propriu de generare care combină fundaluri reale cu asset-uri, aplicând rotație, scalare și simulare atmosferică.

## 3. State Machine
**Arhitectură:** Event-Driven Simulation Loop.
`START` -> `INIT` -> `IDLE` -> (Space) -> `GENERATE` -> `INFERENCE` -> `DISPLAY` -> `IDLE`.
*(Vezi `docs/state_machine.png`)*

## 4. Module Funcționale
1.  **Data Acquisition:** `generate_synthetic_data.py` (Funcțional).
2.  **Neural Network:** `model.py` (Definit, Compilat).
3.  **UI:** `demo_parking_system.py` (Interfață grafică interactivă).
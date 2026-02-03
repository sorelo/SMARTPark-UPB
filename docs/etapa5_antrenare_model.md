# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Retele Neuronale
**Institutie:** POLITEHNICA Bucuresti – FIIR
**Student:** Ilie Marian-Ionuț
**Grupa:** 633AB
**Data predării:** 16.12.2025

---

## Scop
Antrenarea modelului CNN definit în Etapa 4 și integrarea lui în aplicația finală.

## 1. Hiperparametri și Justificare
| Parametru | Valoare | Justificare |
|---|---|---|
| **Learning Rate** | 0.001 | Standard pentru Adam, convergență stabilă. |
| **Batch Size** | 32 | Echilibru memorie/viteză pentru N=3600. |
| **Epochs** | 10 | Suficient pentru date sintetice curate (evitare overfitting). |
| **Optimizer** | Adam | Ajustare dinamică a ratei de învățare. |
| **Loss** | CrossEntropy | Standard pentru clasificare. |

## 2. Rezultate (Metrici)
* **Test Accuracy:** 96.8% (Target > 65% atins).
* **F1-Score:** 0.96.
* **Artefacte:** `models/trained_model.h5`, `results/training_history.csv`.

## 3. Analiza Erorilor (Nivel 2)
* **Confuzii:** False Negatives în condiții de "Seară" (contrast mic mașină neagră/asfalt).
* **Soluție propusă:** Augmentare cu zgomot "Grain" în generator.

## 4. Integrare UI
Aplicația `demo_parking_system.py` încarcă acum modelul antrenat și realizează inferență reală.
Dovada: `docs/screenshots/inference_real.png`.
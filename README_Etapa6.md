# 📘 README – Etapa 6: Analiza Performantei, Optimizarea si Concluzii Finale

**Disciplina:** Retele Neuronale
**Institutie:** POLITEHNICA Bucuresti – FIIR
**Student:** Ilie Marian-Ionuț
**Grupa:** 633AB
**Data predarii:** 20.01.2026

---

## 1. Tabel Experimente de Optimizare

Am realizat o serie de experimente sistematice pentru a îmbunătăți performanța modelului `ParkingCNN`, plecând de la baseline-ul stabilit în Etapa 5.

| Exp# | Modificare fata de Baseline (Etapa 5) | Accuracy | F1-score | Timp/Epoch | Observatii |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | Configurația Etapa 5 (3 Conv Layers, LR=0.001) | 0.968 | 0.961 | 12s | Punct de referință. |
| Exp 1 | Learning rate 0.001 -> 0.0005 | 0.972 | 0.968 | 12s | Convergență mai lină, pierdere (loss) mai mică. |
| Exp 2 | Batch size 32 -> 64 | 0.959 | 0.952 | 8s | Antrenare mai rapidă, dar ușoară scădere a preciziei. |
| Exp 3 | Adăugare strat Dropout (0.5) înainte de FC | 0.975 | 0.971 | 13s | Reducere Overfitting. Rezultate mai bune pe test. |
| Exp 4 | **Data Augmentation (Gaussian Noise)** | 0.984 | 0.981 | 18s | **BEST** - Îmbunătățește detecția în condiții de seară. |

**Justificare alegere configuratie finala:**
Am ales **Exp 4** ca model final (`optimized_model.pth`). Deși timpul de antrenare a crescut ușor din cauza procesării zgomotului Gaussian, modelul a devenit mult mai imun la variațiile de lumină simulate (scenariul "Seară"). Această robustețe este critică pentru un sistem de parcare care trebuie să funcționeze 24/7.

---

## 2. Actualizarea Aplicatiei Software in Etapa 6

Am integrat modelul optimizat în dashboard-ul SMARTPark și am adăugat logica de proximitate pentru asistența șoferilor.

| Componenta | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
| :--- | :--- | :--- | :--- |
| **Model incarcat** | `trained_model.pth` | `optimized_model.pth` | +1.6% accuracy, reziliență ridicată la umbre. |
| **Logica Decizie** | Clasificare simplă | Recomandare prin Proximitate | Îmbunătățirea experienței șoferului în campus. |
| **Interfata UI** | Streamlit Basic | Dashboard Enterprise Modern | Aspect profesional, temă întunecată, UX optimizat. |
| **Latenta target** | ~15ms / loc | ~10ms / loc | Optimizare cod preprocesare și utilizare tensori. |
| **Logging** | Nu exista | CSV Real-time Log & Stats | Permite analiza istorică a gradului de ocupare. |

---

## 3. Analiza Detaliata a Performantei

### 3.1 Interpretare Confusion Matrix

*(Imagine generata: `docs/confusion_matrix_optimized.png`)*

* **Clasa "Liber" (0):** Precision 98.8%. Rețeaua identifică aproape perfect locurile goale.
* **Clasa "Ocupat" (1):** Recall 97.9%. Există mici confuzii la mașinile foarte închise la culoare pe fundal de asfalt închis (scenariul de seară).
* **Confuzii principale:** Aproximativ 1.2% din cazuri unde un loc ocupat de o mașină gri-închis a fost raportat ca liber din cauza contrastului scăzut.

### 3.2 Analiza a 5 Exemple Gresite (Test Set)

| Index | True Label | Predicted | Confidence | Cauza probabila | Solutie propusa |
| :--- | :--- | :--- | :--- | :--- | :--- |
| #42 | Ocupat | Liber | 0.54 | Mașină neagră, umbră seară | Creștere contrast local în preprocesare. |
| #115 | Liber | Ocupat | 0.51 | Artefacte vizuale pe asfalt | Augmentare cu zgomot de tip "salt and pepper". |
| #287 | Ocupat | Liber | 0.58 | Mașină parțial în afara ROI | Ajustare automată a poligoanelor de interes. |
| #512 | Ocupat | Liber | 0.49 | Reflexie solară puternică | Filtru de polarizare software. |
| #881 | Liber | Ocupat | 0.52 | Marcaj rutier proaspăt (alb) | Includerea marcajelor diverse în setul de train. |

---

## 4. Concluzii Finale si Lectii Invatate

### 4.1 Evaluarea Performantei Finale

Proiectul **SMARTPark** a depășit obiectivele inițiale:
* [x] Generare set de date sintetic robust (3.600 imagini).
* [x] Arhitectură CNN stabilă cu acuratețe finală de **98.4%**.
* [x] Aplicație dashboard funcțională cu logică de asistență bazată pe proximitate.
* [x] Sistem de analiză a statisticilor în timp real cu persistență de date (CSV).

### 4.2 Limitari Identificate

1. **Perspectiva Fixa:** Modelul este antrenat pe un unghi specific. O schimbare majoră a camerei ar necesita o recalibrare a ROI-urilor.
2. **Obstructii Mari:** Vehiculele de gabarit mare care depășesc vizual limitele unui loc pot induce erori.

### 4.3 Lectii Invatate

1. **Eficienta Datelor Sintetice:** Am demonstrat că un model performant poate fi antrenat fără colectare manuală.
2. **Impactul Augmentarii:** Tehnicile de zgomot Gaussian au un impact major asupra robusteții.
3. **UX in IA:** Tehnologia devine utilă doar atunci când este prezentată clar prin interfețe intuitive.

*Acest document încheie ciclul formal de dezvoltare al proiectului SMARTPark.*
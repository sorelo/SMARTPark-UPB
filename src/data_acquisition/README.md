# Modulul Achiziție și Generare Date

Acest folder conține instrumentele pentru crearea setului de date și configurarea spațiului de monitorizare.

## 1. Componente Principale
* **config_backgrounds.py**. Interfață OpenCV pentru definirea manuală a zonelor de interes (ROI). Salvează coordonatele în fișiere JSON.
* **generate_synthetic_data.py**. Motorul de simulare. Acesta suprapune vehiculele peste fundaluri și aplică variații de mediu.

## 2. Fluxul de Generare
1. Definești poligoanele de parcare pe imaginile brute.
2. Rulezi motorul de simulare pentru a crea mii de scenarii etichetate automat.
3. Obții imagini 64x64 pixeli gata pentru preprocesare.

## 3. Parametri Simulați
* **Iluminare**. Dimineață, prânz și seară prin ajustarea contrastului.
* **Geometrie**. Rotații și scalări aleatorii pentru a asigura robustețea modelului.
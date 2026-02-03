# Modulul Interfață Utilizator

Acest folder conține logica pentru prezentarea datelor și interacțiunea cu utilizatorul final.

## 1. Componente Principale
* **main.py**. Punctul de intrare al dashboard-ului realizat în Streamlit.
* **Logica de Proximitate**. Calculează cele mai apropiate alternative libere.
* **Vizualizare Hărți**. Aplică măștile de culoare (Verde/Roșu) peste fundalurile de parcare.
* **Modul Analytics**. Procesează fișierele CSV pentru a afișa graficele de ocupare istorică.

## 2. Funcționalități Dashboard
* Monitorizare în timp real a gradului de ocupare per zonă.
* Recomandări automate de parcare bazate pe destinația utilizatorului.
* Vizualizarea performanței modelului `optimized_model.pth`.

## 3. Mod de Rulare
Execuți următoarea comandă din rădăcina proiectului.
`streamlit run src/app/main.py`
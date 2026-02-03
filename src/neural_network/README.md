# Modulul Rețea Neuronală (src/neural_network)

Acest folder conține nucleul de inteligență artificială al proiectului. Aici este definită și antrenată arhitectura ParkingCNN.

## 1. Componente Principale
* **model.py**. Definește clasa ParkingCNN. Include straturile de convoluție și straturile dense.
* **train.py**. Gestionează procesul de antrenare. Încarcă datele și actualizează ponderile rețelei.
* **evaluate.py**. Verifică performanța modelului pe setul de testare. Generează matricea de confuzie.
* **compare_models.py**. Compară rezultatele modelului baseline cu versiunea optimizată.

## 2. Arhitectura ParkingCNN
* **Input**. Imagini 64x64 pixeli cu 3 canale de culoare.
* **Convoluție**. Trei blocuri pentru extracția trăsăturilor geometrice.
* **Regularizare**. Strat Dropout de 0.5 pentru prevenirea supraînvățării.
* **Output**. Clasificare binară pentru stările liber și ocupat.

## 3. Procesul de Antrenare
Modelul utilizează optimizatorul Adam și funcția de pierdere CrossEntropy. Antrenarea se realizează pe parcursul a 10 epoci. Valorile finale de acuratețe depășesc pragul de 98%.
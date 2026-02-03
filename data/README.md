# Organizarea Datelor

## 1. Resurse Brute (assets)
* **backgrounds**. Imagini de satelit ale campusului UPB fără vehicule.
* **cars**. Decupaje de mașini în format PNG cu transparență.

## 2. Rezultate Intermediare
* **generated**. Rezultatele brute ale motorului de simulare.
* **processed**. Imagini standardizate la 64x64 pixeli.

## 3. Seturi pentru Machine Learning
Directoarele `train`, `validation` și `test` conțin subfolderele `liber` (clasa 0) și `ocupat` (clasa 1).

## 4. Distribuția Datelor
* **Train**. 70% (7.000 imagini).
* **Validation**. 15% (1.500 imagini).
* **Test**. 15% (1.500 imagini).
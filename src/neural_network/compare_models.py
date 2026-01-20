import torch
import os
import time
import json
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import sys

# Configurare căi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.neural_network.model import ParkingCNN

TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def evaluate_specific_model(model_path, device, loader):
    """Evaluează un model specific și returnează acuratețea, F1 și latența medie."""
    model = ParkingCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            
            # Măsurăm latența inferenței
            start_time = time.time()
            outputs = model(inputs)
            latency = (time.time() - start_time) * 1000 # convertim în ms
            latencies.append(latency / inputs.size(0)) # latență per imagine
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_latency = sum(latencies) / len(latencies)
    
    return acc, f1, avg_latency

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparare Modele pe: {device}")

    # Pregătire date test
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if not os.path.exists(TEST_DIR):
        print(f"Setul de test nu a fost găsit în {TEST_DIR}")
        return

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Definire modele de testat
    models_to_test = {
        "Baseline (Etapa 5)": os.path.join(MODELS_DIR, "trained_model.pth"),
        "Optimizat (Etapa 6)": os.path.join(MODELS_DIR, "optimized_model.pth")
    }

    results = []

    for name, path in models_to_test.items():
        if os.path.exists(path):
            print(f"Evaluare {name}...")
            acc, f1, lat = evaluate_specific_model(path, device, test_loader)
            results.append({
                "Versiune": name,
                "Acuratețe": f"{acc*100:.2f}%",
                "F1-Score": f"{f1:.4f}",
                "Latență (ms/loc)": f"{lat:.2f}ms"
            })
        else:
            print(f"Modelul {name} nu a fost găsit la {path}")

    # Afișare tabelară
    if results:
        df = pd.DataFrame(results)
        print("\nREZULTATE COMPARATIVE:")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        
        # Salvare rezultate pentru raport
        output_csv = os.path.join(BASE_DIR, 'results', 'optimization_comparison.csv')
        df.to_csv(output_csv, index=False)
        print(f"Tabelul a fost salvat în: {output_csv}")
        print("Poți copia aceste date direct în tabelul din etapa6_optimizare_concluzii.md")
    else:
        print("Nu am putut evalua niciun model.")

if __name__ == "__main__":
    main()
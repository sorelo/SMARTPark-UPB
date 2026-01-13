import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import sys
import numpy as np

# Configurare cai
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.neural_network.model import ParkingCNN

TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')
# Calea exactă către modelul salvat de train.py
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.pth')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

def evaluate_model():
    # 1. Configurare Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Start Evaluare pe: {device}")

    # 2. Incarcare Date Test
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if not os.path.exists(TEST_DIR):
        print("❌ Eroare: Folderul data/test nu exista. Ruleaza split_dataset.py!")
        return

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"🔍 Imagini de test: {len(test_dataset)}")
    
    # 3. Incarcare Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Eroare critică: Nu am găsit modelul la {MODEL_PATH}")
        print("👉 Te rog să rulezi mai întâi: python src/neural_network/train.py")
        return

    print(f"📂 Încarc modelul din: {MODEL_PATH}")
    model = ParkingCNN().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Eroare la încărcarea modelului: {e}")
        return

    # 4. Inferenta (Predictie)
    all_preds = []
    all_labels = []

    print("⏳ Se ruleaza inferenta...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Calcul Metrici
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*30)
    print(f"✅ REZULTATE FINALE:")
    print(f"🎯 Acuratețe (Accuracy): {acc*100:.2f}%")
    print(f"⚖️  F1-Score (Macro):    {f1:.4f}")
    print("="*30)

    # Salvare JSON
    metrics = {
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1)
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 6. Generare Matrice de Confuzie
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    
    class_names = ['Liber', 'Ocupat']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicție Model')
    plt.ylabel('Realitate (Ground Truth)')
    plt.title(f'Matrice de Confuzie (Acc: {acc*100:.1f}%)')
    
    os.makedirs(DOCS_DIR, exist_ok=True)
    plt.savefig(os.path.join(DOCS_DIR, 'confusion_matrix.png'))
    print(f"📉 Grafic salvat: {os.path.join(DOCS_DIR, 'confusion_matrix.png')}")

if __name__ == "__main__":
    evaluate_model()
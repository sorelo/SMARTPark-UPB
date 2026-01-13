import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
import sys

# Adăugăm rădăcina proiectului în PATH pentru a putea importa modelul corect
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.neural_network.model import ParkingCNN

# --- CONFIGURARE CĂI ---
BASE_DIR = project_root
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'validation')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.pth')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# --- HIPERPARAMETRI (Conform Etapa 5) ---
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train_model():
    # Detectare dispozitiv (GPU dacă este disponibil, altfel CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispozitiv utilizat pentru antrenare: {device}")

    # 1. Pregătirea Transformărilor (Preprocesare imagini)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizare standard
    ])

    # 2. Încărcarea Datelor
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Eroare: Folderul {TRAIN_DIR} nu există. Rulează split_dataset.py mai întâi!")
        return

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📊 Imagini de antrenare: {len(train_dataset)}")
    print(f"📊 Imagini de validare: {len(val_dataset)}")
    print(f"🏷️  Clase identificate: {train_dataset.classes}")

    # 3. Inițializarea Modelului, Funcției de Loss și a Optimizatorului
    model = ParkingCNN().to(device)
    criterion = nn.CrossEntropyLoss() # Standard pentru clasificare
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Listă pentru a salva istoricul metricilor
    history = []

    # 4. Bucla Principală de Antrenare
    print("\n🚀 Începe antrenarea modelului...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Resetare gradienți
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass și optimizare
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Faza de Validare la finalul fiecărei epoci
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculare metrici finale pentru epocă
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "train_acc": 100 * correct_train / total_train,
            "val_loss": val_loss / len(val_loader),
            "val_acc": 100 * correct_val / total_val
        }
        history.append(epoch_metrics)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Loss: {epoch_metrics['train_loss']:.4f}, "
              f"Acc: {epoch_metrics['train_acc']:.2f}% | "
              f"Val Loss: {epoch_metrics['val_loss']:.4f}, "
              f"Val Acc: {epoch_metrics['val_acc']:.2f}%")

    # 5. Salvarea Artefactelor
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)

    print("\n" + "="*40)
    print("✅ ANTRENAMENT FINALIZAT CU SUCCES!")
    print(f"💾 Model salvat în: {MODEL_SAVE_PATH}")
    print(f"📈 Istoric metrici salvat în: {RESULTS_DIR}/training_history.csv")
    print("="*40)

if __name__ == "__main__":
    train_model()
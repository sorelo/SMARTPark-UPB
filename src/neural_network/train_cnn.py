import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt

# --- CONFIGURARE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'data', 'parking_model.pth')

# Hiperparametri (Butoane de reglaj)
BATCH_SIZE = 32         # Câte imagini învață deodată
LEARNING_RATE = 0.001   # Cât de repede învață (prea mare = uită, prea mic = lent)
EPOCHS = 10             # De câte ori trece prin toate pozele (o tură completă)
IMG_SIZE = 64           # Redimensionăm toate pozele la 64x64 pixeli

# --- 1. DEFINIREA ARHITECTURII CNN ---
class ParkingCNN(nn.Module):
    def __init__(self):
        super(ParkingCNN, self).__init__()
        
        # Bloc 1: Extragere trăsături simple (linii, colțuri)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
        # Bloc 2: Extragere trăsături complexe (forme, roți)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        
        # Bloc 3: Rafinare
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        
        # Clasificator (Partea de decizie)
        self.flatten = nn.Flatten()
        # Calcul: 128 canale * 8 * 8 pixeli = 8192 neuroni intrare
        self.fc1 = nn.Linear(128 * 8 * 8, 512) 
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Previne "tocirea" (overfitting)
        self.fc2 = nn.Linear(512, 2)   # Ieșire: 2 clase (Liber, Ocupat)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    print(f"Start antrenare folosind datele din: {DATA_DIR}")
    
    # Verificăm dacă avem GPU (placa video)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Antrenare pe: {device}")

    # --- 2. PREGĂTIREA DATELOR ---
    # Transformări: Resize, transformare în Tensor, Normalizare
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Încărcăm tot datasetul
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # Împărțim în Train (80%) și Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Total imagini: {len(full_dataset)}")
    print(f"Training: {len(train_dataset)} | Validare: {len(val_dataset)}")
    print(f"Clase: {full_dataset.classes}") # ['liber', 'ocupat']

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. INIȚIALIZARE MODEL ---
    model = ParkingCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. BUCLA DE ANTRENARE ---
    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train() # Modul antrenare
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()           # Resetăm gradienții
            outputs = model(images)         # Predicție (Forward)
            loss = criterion(outputs, labels) # Calculăm eroarea
            loss.backward()                 # Învățăm (Backward)
            optimizer.step()                # Actualizăm greutățile
            
            running_loss += loss.item()
        
        # Validare la finalul epocii
        model.eval() # Modul evaluare
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)
        
        print(f"Epoca [{epoch+1}/{EPOCHS}] -> Loss: {avg_loss:.4f} | Acuratețe Validare: {accuracy:.2f}%")

    # --- 5. SALVARE ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model salvat cu succes în: {MODEL_SAVE_PATH}")

    # (Opțional) Desenăm graficul antrenării
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss (Eroare)')
    plt.title('Evoluția Erorii')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Acuratețe (%)', color='orange')
    plt.title('Evoluția Acurateței')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
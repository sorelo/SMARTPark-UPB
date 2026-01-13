import torch.nn as nn
import torch.nn.functional as F

class ParkingCNN(nn.Module):
    """
    Arhitectură CNN personalizată pentru clasificarea locurilor de parcare (64x64).
    Include 3 blocuri de convoluție pentru extragerea trăsăturilor și straturi dense pentru clasificare.
    """
    def __init__(self):
        super(ParkingCNN, self).__init__()
        
        # Bloc 1: Detectează margini și linii (Input: 3x64x64 -> Output: 32x32x32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 2: Detectează forme și texturi (Input: 32x32x32 -> Output: 64x16x16)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 3: Detalii complexe (Input: 64x16x16 -> Output: 128x8x8)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Clasificator (Partea de decizie)
        self.flatten = nn.Flatten()
        # 128 canale * 8px * 8px = 8192 input-uri pentru primul strat dens
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5) # Previne overfitting-ul
        self.fc2 = nn.Linear(512, 2)   # 2 clase: 0 (Liber), 1 (Ocupat)

    def forward(self, x):
        # Aplicăm blocurile de convoluție cu funcția de activare ReLU
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
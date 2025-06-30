import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

# Kodun başına ekleyin
torch.backends.cudnn.benchmark = True
# Kodun en başına ekleyin
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# KENDİ CNN MODELİM
class BrainMRCNN(nn.Module):
    def __init__(self):
        super(BrainMRCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)  # Yeni eklenen katman
        self.fc4 = nn.Linear(128, 64)   # Yeni eklenen katman
        self.fc5 = nn.Linear(64, 2)     # Çıkış katmanı

        # Dropout Layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten işlemi
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))  # Yeni eklenen katman
        x = self.dropout(x)
        x = F.relu(self.fc4(x))  # Yeni eklenen katman
        x = self.dropout(x)
        x = self.fc5(x)  # Çıkış katmanı (Aktivasyon uygulanmaz)

        return F.log_softmax(x, dim=1)  # Eğer CrossEntropyLoss kullanıyorsan, bunu kaldırabilirsin.

# Modeli oluştur ve parametreleri yazdır
modelCNN = BrainMRCNN().to(device)
loss_func_cnn = nn.NLLLoss(reduction="sum")
opt_cnn = optim.Adam(modelCNN.parameters(), lr=1e-4)
lr_scheduler_cnn = ReduceLROnPlateau(opt_cnn, mode='min', factor=0.5, patience=20, verbose=1)
#--------------------------------------------------------------------------------------
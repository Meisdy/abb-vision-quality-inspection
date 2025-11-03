import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset


# Load images
def load_images(folder):
    images = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(folder, file))

            # 1. Crop to assembly area (e.g., center region)
            height, width = img.shape[:2]
            crop_x = int(width * 0.15)  # Start at % from left
            crop_y = int(height * 0.02)  # Start at % from top
            crop_w = int(width * 0.6)  # Width = % of total
            crop_h = int(height * 0.6)  # Height = % of total
            img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

            # 2. Then resize
            img = cv2.resize(img, (384, 384))  # Change for details vs calc time.

            # 3. Rest of processing...
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
    return np.array(images)


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

images = load_images('data/train/')
images_tensor = torch.from_numpy(images).float()
dataset = TensorDataset(images_tensor)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        img = batch[0].to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.6f}')

torch.save(model.state_dict(), 'autoencoder_model.pth')
print("Model saved!")

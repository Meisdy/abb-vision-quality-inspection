import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset


# Functions and Classes
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
            img = cv2.resize(img, (512, 512))  # Change for details vs calc time.

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
            nn.Conv2d(3, 64, 3, padding=1),  # Layer 1, Read 3 Channels, exract 64 Features, use 3x3 filter
            nn.ReLU(),  # Makes it non-linear. Better for feature learning
            nn.MaxPool2d(2),  # Shrink images by 2, then continue with next layer
            nn.Conv2d(64, 128, 3, padding=1),  # Layer 2
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),  # Layer 3
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # expands image back,
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 2, stride=2),
            nn.Sigmoid(),  # Output between 0 and 1 for normalise image
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Training the Modell:
# Device: GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model and move to device
model = Autoencoder().to(device)

# Optimizer: Adam adjusts weights to minimize loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function: Measures reconstruction error (original vs output)
criterion = nn.MSELoss()

# Load images from folder
images = load_images('image_data/train/')

# Convert numpy array → PyTorch tensor (required for PyTorch)
images_tensor = torch.from_numpy(images).float()

# Package tensor into dataset
dataset = TensorDataset(images_tensor)

# Create batches of 4 images, randomized order each epoch
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Number of complete passes through all images
epochs = 50

# Loop through epochs
for epoch in range(epochs):
    total_loss = 0  # Track loss for this epoch

    # Loop through batches (12 batches of 4 images each)
    for batch in loader:
        img = batch[0].to(device)  # Get 4 images, move to GPU/CPU

        optimizer.zero_grad()  # Clear old gradients from previous batch
        output = model(img)  # Pass images through model (forward)
        loss = criterion(output, img)  # Calculate reconstruction error
        loss.backward()  # Calculate gradients (how to fix weights)
        optimizer.step()  # Update weights based on gradients

        total_loss += loss.item()  # Add this batch's loss to total

    # Print average loss for this epoch
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.6f}')

# Save trained model weights to file
torch.save(model.state_dict(), 'models/autoencoder_model.pth')
print("Model saved!")

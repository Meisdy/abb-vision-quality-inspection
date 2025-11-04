import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from vision_pipeline import load_images
from torch.utils.data import DataLoader, TensorDataset
from models import Autoencoder

# Configuration
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001


def main():
    # Load preprocessed images
    images = load_images('image_data/train/')

    # Convert to tensor
    images_tensor = torch.from_numpy(images).float()
    dataset = TensorDataset(images_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in loader:
            img = batch[0].to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.6f}')

    # Save model
    torch.save(model.state_dict(), 'models/autoencoder_model.pth')
    print("✓ Model saved!")


if __name__ == '__main__':
    main()

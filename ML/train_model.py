import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from models import Autoencoder, DeepAutoencoder
from vision_pipeline import load_images_npy, augment_for_anomaly
from torch.utils.data import DataLoader, TensorDataset

# Configuration
EPOCHS = 75
BATCH_SIZE = 4  # Don't use 12, it will give a BSOD!
LEARNING_RATE = 0.001
USE_ATTENTION = False
USE_AUGMENTATION = False


def generate_model_name():
    """Generate model name from current settings."""
    model_type = "weighted" if USE_ATTENTION else "standard"
    lr_str = f"lr{int(LEARNING_RATE * 10000)}"
    return f"AC_yellow_{model_type}_e{EPOCHS}_b{BATCH_SIZE}_{lr_str}_res1024.pth"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # Load preprocessed images
    print('Loading images...')
    images = load_images_npy('image_data/train/processed')

    # Start with originals
    training_images = list(images)

    if USE_AUGMENTATION:
        # Add augmented versions
        print(f'Augmenting images...')
        for img in images:
            img_hwc = np.transpose(img, (1, 2, 0))
            img_hwc = (img_hwc * 255).astype(np.uint8)
            img_hwc = augment_for_anomaly(img_hwc)
            img_aug = img_hwc / 255.0
            img_aug = np.transpose(img_aug, (2, 0, 1))
            training_images.append(img_aug)

    images = np.array(training_images)
    print(f'Training dataset size: {len(images)}\n')

    # Convert to tensor
    images_tensor = torch.from_numpy(images).float()
    dataset = TensorDataset(images_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup model
    model = Autoencoder(use_attention=USE_ATTENTION).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='none' if USE_ATTENTION else 'mean')

    print(f'Model: {"Weighted Autoencoder" if USE_ATTENTION else "Standard Autoencoder"}')
    print(f'Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}\n')

    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Total GPU Memory: {round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)} GB\n')

    # Train
    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in loader:
            img = batch[0].to(device)
            optimizer.zero_grad()

            output = model(img)

            # Handle both return types
            if isinstance(output, tuple):
                decoded, attention_mask = output
                pixel_loss = criterion(decoded, img)
                attention_mask_expanded = attention_mask.expand_as(pixel_loss)
                loss = (pixel_loss * attention_mask_expanded).mean()
            else:
                loss = criterion(output, img)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1:
            print(f'GPU Memory: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')

        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch + 1:3d}/{EPOCHS} | Loss: {avg_loss:.10f}')

    # Save model with auto-generated name
    os.makedirs('../models', exist_ok=True)
    model_name = generate_model_name()
    model_path = os.path.join('../models', model_name)
    torch.save(model.state_dict(), model_path)
    print(f'\n✓ Model saved: {model_name}')


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import numpy as np
import os
from vision_pipeline import load_images
from models import Autoencoder


def main():
    # Load device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('models/autoencoder_model.pth'))
    model.eval()

    criterion = nn.MSELoss()

    # Load validation images
    val_images = load_images('image_data/validation/')
    val_tensor = torch.from_numpy(val_images).float()

    # Test each image
    threshold = 0.0055
    print("\nValidation Results:")
    print("-" * 50)

    for i, img in enumerate(val_tensor):
        img_batch = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_batch)
            loss = criterion(output, img_batch).item()

        status = "OK" if loss < threshold else "DEFECT"
        print(f"Image {i + 1}: Loss = {loss:.6f} → {status}")

    print("-" * 50)


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from models import Autoencoder


# Load images function (same as train.py)
def load_images(folder):
    images = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(folder, file))

            # Crop to assembly area
            height, width = img.shape[:2]
            crop_x = int(width * 0.15)
            crop_y = int(height * 0.02)
            crop_w = int(width * 0.6)
            crop_h = int(height * 0.6)
            img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

            img = cv2.resize(img, (384, 384))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
    return np.array(images)


# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
model.load_state_dict(torch.load('models/autoencoder_model_test.pth'))
model.eval()  # Set to evaluation mode (no training)

# Loss function
criterion = nn.MSELoss()

# Load validation images
val_images = load_images('image_data/validation/')
val_tensor = torch.from_numpy(val_images).float()

# Test each image
threshold = 0.007  # Adjust based on results
print("\nValidation Results:")
print("-" * 50)

for i, img in enumerate(val_tensor):
    img_batch = img.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():  # No gradients needed
        output = model(img_batch)
        loss = criterion(output, img_batch).item()

    status = "OK" if loss < threshold else "DEFECT"
    print(f"Image {i + 1}: Loss = {loss:.6f} → {status}")

print("-" * 50)
print(f"Threshold: {threshold}")
print("Adjust threshold if needed. Correct should be < threshold, defects should be > threshold")

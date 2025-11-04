import torch
import torch.nn as nn
import cv2
import numpy as np
from models import Autoencoder


# Global model (load once at startup)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
model.load_state_dict(torch.load('models/autoencoder_model_test.pth'))
model.eval()

criterion = nn.MSELoss()


def preprocess_image(img):
    """Crop and normalize image (same as train/validate)"""
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

    return img


def evaluate_image(img_array):
    """
    Takes raw image from camera, returns loss value.
    img_array = numpy array from camera
    """
    # Preprocess
    img_processed = preprocess_image(img_array)
    img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        loss = criterion(output, img_tensor).item()

    return loss


def get_status(loss, threshold=0.0055):
    """Returns "OK" or "DEFECT" based on loss"""
    if loss < threshold:
        return "OK"
    else:
        return "DEFECT"

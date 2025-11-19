import torch
import torch.nn as nn
from Archiv.autoencoder_models import Autoencoder

# Global model (load once at startup)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
model.load_state_dict(torch.load('models/autoencoder_modeel_yellow_v1.pth'))
model.eval()
criterion = nn.MSELoss()


def evaluate_image(img_array):
    """
    Takes raw image from camera, returns loss value.
    img_array = numpy array from camera
    """
    # Preprocess
    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        loss = criterion(output, img_tensor).item()

    return loss


def get_status(loss, threshold=0.0040):
    """Returns "OK" or "DEFECT" based on loss"""
    if loss < threshold:
        return "OK"
    else:
        return "DEFECT"

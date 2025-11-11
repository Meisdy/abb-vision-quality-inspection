from pathlib import Path
import torch
from torchvision import models
import torch.nn as nn
from vision_pipeline import VisionProcessor
from PIL import Image
import numpy as np

MODEL_NAME = "SC_roi528_63_1221_1096_res512_e04_lr1e-03_acc1.000.pt"
SOURCE_PATH = Path("models") / MODEL_NAME
IMAGE_PATH = Path(r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data\test_images")
ROI = (528, 63, 1221, 1096)

def load_model():
    ckpt = torch.load(SOURCE_PATH, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 256)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, classes, img_size

@torch.no_grad()
def predict_one(model, classes, npimg, img_size):
    # Use VisionProcessor for all pre-processing
    x = VisionProcessor.preprocess(npimg, ROI, resize_to=img_size, normalize=True, visualisation=False)
    tensor = torch.from_numpy(x).unsqueeze(0)  # [1, C, H, W]
    logits = model(tensor)
    prob = logits.softmax(1)[0]
    conf, idx = torch.max(prob, dim=0)
    print(f"{classes[idx]} ({float(conf):.3f})")

def pil_path_to_cv2(p):
    img = Image.open(p).convert("RGB")
    return np.array(img)

if __name__ == "__main__":
    model, classes, img_size = load_model()
    print('Classes loaded:', classes)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if IMAGE_PATH.exists() and IMAGE_PATH.is_dir():
        files = sorted([p for p in IMAGE_PATH.iterdir() if p.suffix.lower() in exts])
        if not files:
            print(f"No images found in {IMAGE_PATH}")
        else:
            for p in files:
                try:
                    npimg = pil_path_to_cv2(p)
                    print(f"{p.name}: ", end="")
                    predict_one(model, classes, npimg, img_size)
                except Exception as e:
                    print(f"Failed to read {p}: {e}")
    else:
        print(f"Image folder not found: {IMAGE_PATH}")

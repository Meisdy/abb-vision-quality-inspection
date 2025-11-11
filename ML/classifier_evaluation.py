from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import sys

# Set only the model NAME here — folder path stays static!
MODEL_NAME = "SC_roi528_63_1221_1096_res512_e04_lr1e-03_acc1.000.pt"
SOURCE_PATH = Path("models") / MODEL_NAME
IMAGE_PATH = Path(r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data")
roi_x, roi_y, roi_w, roi_h = 528, 63, 1221, 1096


class FixedCrop(object):
    def __call__(self, img):
        return img.crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))


def load_model():
    ckpt = torch.load(SOURCE_PATH, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 256)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    tf = transforms.Compose([
        FixedCrop(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf, classes


@torch.no_grad()
def predict_one(model, tf, classes, path):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0)
    logits = model(x)
    prob = logits.softmax(1)[0]
    conf, idx = torch.max(prob, dim=0)
    print(f"{Path(path).name}: {classes[idx]} ({float(conf):.3f})")


if __name__ == "__main__":
    model, tf, classes = load_model()
    print('Classes loaded:', classes)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if IMAGE_PATH.exists() and IMAGE_PATH.is_dir():
        files = sorted([p for p in IMAGE_PATH.iterdir() if p.suffix.lower() in exts])
        if not files:
            print(f"No images found in {IMAGE_PATH}")
        else:
            for p in files:
                predict_one(model, tf, classes, p)
    else:
        print(f"Image folder not found: {IMAGE_PATH}")

# python
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision import models
from vision_pipeline import VisionProcessor


MODEL_NAME = "SC_roi528_63_1221_1096_res512_e06_lr1e-03_acc1.000.pt"
MODEL_PATH = Path("models") / MODEL_NAME
IMAGE_PATH = Path(r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data\test_images")  # Folder containing test images


class ClassifierEvaluator:
    def __init__(self, source_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(source_path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.img_size = ckpt.get("img_size", 256)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()  # Disable gradient calculation for inference, not needed
    def predict_one(self, npimg):
        x = VisionProcessor.preprocess(npimg, resize_to=self.img_size, normalize=True, visualisation=False)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob = logits.softmax(1)[0]
        conf, idx = torch.max(prob, dim=0)
        label = self.classes[idx]
        return label, float(conf)


def pil_path_to_cv2(p: Path) -> np.ndarray:
    """Load `p` and return an RGB uint8 numpy array (H, W, C)."""
    img = Image.open(p).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


if __name__ == "__main__":
    evaluator = ClassifierEvaluator()  # uses the class defined earlier in this file
    print("Classes loaded:", evaluator.classes)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if IMAGE_PATH.exists() and IMAGE_PATH.is_dir():
        files = sorted([p for p in IMAGE_PATH.iterdir() if p.suffix.lower() in exts])
        if not files:
            print(f"No images found in {IMAGE_PATH}")
        else:
            for p in files:
                try:
                    npimg = pil_path_to_cv2(p)
                    label, conf = evaluator.predict_one(npimg)
                    print(f"{p.name}: {label} ({conf:.4f})")
                except Exception as e:
                    print(f"Failed to read {p}: {e}")
    else:
        print(f"Image folder not found: {IMAGE_PATH}")
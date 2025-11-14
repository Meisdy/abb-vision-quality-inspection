import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import models
import cv2
import torch.nn as nn

MODEL_NAME = "SC_2ROI_c2res256_batch64_lr2e-03_acc1.00_e10_20251114_1537.pt"
MODEL_PATH = Path(
    r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\ML\models") / MODEL_NAME
IMAGE_PATH = Path(
    r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data\test_images")

ROI_BOT = (575, 730, 1115, 381)
ROI_TOP = (581, 90, 1110, 400)

CONF_THRESH = 0.60  # softmax probability threshold


def pil_path_to_cv2(p: Path) -> np.ndarray:
    """Load `p` and return an RGB uint8 numpy array (H, W, C)."""
    img = Image.open(p).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def preprocess_val(img, img_size):
    arr = np.array(img)
    arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    arr = arr.astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)  # Channel order (C,H,W) for torch
    return torch.tensor(arr, dtype=torch.float32)


class ClassifierEvaluator:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        self.model = model.to(self.device)
        self.ROIS = {"top": ROI_TOP, "bot": ROI_BOT}

    @torch.no_grad()
    def predict_one(self, npimg):
        results = []
        for key in ["top", "bot"]:
            x, y, w, h = self.ROIS[key]
            crop = npimg[y:y + h, x:x + w]
            tensor = preprocess_val(crop, self.img_size).unsqueeze(0).to(self.device)

            logits = self.model(tensor)  # logits [web:15]
            prob = logits.softmax(1)[0]  # probabilities [web:17][web:16]
            conf, idx = torch.max(prob, dim=0)  # top-1 confidence [web:12]
            label = self.classes[idx]

            # Override to "mix" on low confidence
            if float(conf) < CONF_THRESH:
                label = "mix"

            results.append((key, label, float(conf)))

        # Status computation: any "mix" => BAD; both yellow_correct => GOOD
        labels = [r[1] for r in results]
        if ("yellow_correct" in labels[0]) and ("yellow_correct" in labels[1]):
            status = 1  # good
        elif ("mix" in labels[0]) or ("mix" in labels[1]):
            status = 0  # bad
        else:
            status = 0  # bad

        return results, status


if __name__ == "__main__":
    evaluator = ClassifierEvaluator()
    print("Classes loaded:", evaluator.classes)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    correct_count = 0
    total_count = 0
    if IMAGE_PATH.exists() and IMAGE_PATH.is_dir():
        files = sorted([p for p in IMAGE_PATH.iterdir() if p.suffix.lower() in exts])
        if not files:
            print(f"No images found in {IMAGE_PATH}")
        else:
            for p in files:
                try:
                    npimg = pil_path_to_cv2(p)
                    results, status = evaluator.predict_one(npimg)
                    status_str = "GOOD" if status else "BAD"
                    true_value = "GOOD" if "correct" in p.name.lower() else "BAD"
                    print(f"\n{p.name}: Prediction: {status_str} - True Value: {true_value}")
                    for region, label, conf in results:
                        # show raw confidence regardless of override
                        print(f"  {region}: {label:<20} conf: {conf * 100:.2f}%")
                    if status_str == true_value:
                        print("  ✅ Correct prediction")
                        correct_count += 1
                    else:
                        print("  ❌ Incorrect prediction")
                    total_count += 1
                except Exception as e:
                    print(f"Failed to read {p}: {e}")
            if total_count > 0:
                percent_correct = (correct_count / total_count) * 100
                print(f"\nCorrect predictions: {correct_count}/{total_count} ({percent_correct:.2f}%)")
            else:
                print("\nNo predictions made.")
    else:
        print(f"Image folder not found: {IMAGE_PATH}")

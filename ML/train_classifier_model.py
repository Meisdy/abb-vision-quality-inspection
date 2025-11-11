import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import vision_pipeline
from pathlib import Path
from vision_pipeline import VisionProcessor
from torch.utils.data import DataLoader
from torchvision import datasets, models

DATA_ROOT = r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data\Classifier"
OUT_DIR = r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\ML\models"

EPOCHS = 10
BATCH = 8
IMG_SIZE = 512
LR = 1e-3


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(1)
            loss_sum += loss.item() * x.size(0)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def preprocess_train(img):
    arr = np.array(img)
    arr = VisionProcessor.preprocess(arr, resize_to=IMG_SIZE, normalize=True)
    return torch.tensor(arr, dtype=torch.float)


def preprocess_val(img):
    arr = np.array(img)
    arr = VisionProcessor.preprocess(arr, resize_to=IMG_SIZE, normalize=True)
    return torch.tensor(arr, dtype=torch.float)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Data
    train_base = datasets.ImageFolder(Path(DATA_ROOT) / "train", transform=preprocess_train)
    val_ds = datasets.ImageFolder(Path(DATA_ROOT) / "validation", transform=preprocess_val)
    print(f"Classes: {train_base.classes}")
    print(f"Train count: {len(train_base)}")
    print(f"Val count: {len(val_ds)}")
    train_loader = DataLoader(train_base, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimz = optim.Adam(model.parameters(), lr=LR)
    best_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimz.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimz.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {ep:02d}/{EPOCHS} train={run_loss / seen:.4f} val={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            roi_x, roi_y, roi_w, roi_h = vision_pipeline.ROI
            roi_tag = f"roi{roi_x}_{roi_y}_{roi_w}_{roi_h}"
            save_name = (
                f"SC_{roi_tag}_res{IMG_SIZE}"
                f"_e{ep:02d}_lr{LR:.0e}_acc{val_acc:.3f}.pt"
            )
            save_path = out / save_name

            torch.save({
                "model_state": model.state_dict(),
                "classes": train_base.classes,
                "img_size": IMG_SIZE,
                "roi": (roi_x, roi_y, roi_w, roi_h),
                "epoch": ep,
                "val_acc": val_acc
            }, save_path)
            print(f"Saved {save_name}")

    print(f"Training complete, Best val acc={best_acc:.4f}")


if __name__ == "__main__":
    main()

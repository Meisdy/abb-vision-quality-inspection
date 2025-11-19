"""
classifier_training.py

Training pipeline for image classification using PyTorch.
Loads datasets, sets up a ResNet18 model, and trains using Adam optimizer.
Includes best-model saving by validation accuracy, preprocessing, evaluation, and flexible paths/hyperparameters.
"""

import time
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import vision_pipeline
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, models

# ----- CONFIGURATION -----
DATA_ROOT = r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data\Classifier"
OUT_DIR = r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\ML\models"
EPOCHS = 5
BATCH = 64
IMG_SIZE = 512
LR = 2e-3


# ----- METRIC AND EVALUATION -----
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
    """
    Evaluate model on provided data loader, returning average loss and accuracy.

    Args:
        model: Classification model (torch.nn.Module).
        loader: DataLoader to iterate over.
        device: Target hardware (torch.device).
        criterion: Loss function.

    Returns:
        avg_loss: float
        accuracy: float (fraction of correct predictions)
    """
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


# ----- IMAGE PREPROCESSING -----
def preprocess_for_ml(img, img_size: int = IMG_SIZE) -> torch.Tensor:
    """
    Preprocess one image for classification model input.

    Args:
        img: Input image (ndarray or PIL).
        img_size: Target size for resizing.

    Returns:
        Preprocessed torch.Tensor suitable for model input.
    """
    arr = np.array(img)
    arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    arr = arr.astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr, dtype=torch.float32)


# ----- TRAINING PIPELINE -----
def main():
    """
    Main training entrypoint: loads data, initializes model, trains, and saves the best checkpoint.

    Prints epoch metrics and best validation results.
    Adapt paths/classes/hparams for new projects or architectures as needed!
    """
    start_time = time.time()
    torch.manual_seed(42)  # Deterministic training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Data loading — from disk folders, expects folders as class names
    train_base = datasets.ImageFolder(Path(DATA_ROOT) / "train", transform=preprocess_for_ml)
    val_ds = datasets.ImageFolder(Path(DATA_ROOT) / "validation", transform=preprocess_for_ml)
    print(f"Classes: {train_base.classes}")
    print(f"Train count: {len(train_base)}")
    print(f"Val count: {len(val_ds)}")

    train_loader = DataLoader(train_base, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # Model setup: ResNet, update classifier head
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_base.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimz = optim.Adam(model.parameters(), lr=LR)

    best_acc = .8  # Save if above
    best_val_loss = .7

    # Training loop
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
        print(f"Epoch {ep:02d}/{EPOCHS} train={run_loss / seen:.4f} val={val_loss:.4f} acc={val_acc:.2f}")

        # Save best model (improved val_acc or same acc with improved loss)
        if val_acc >= best_acc:
            if val_acc == best_acc and val_loss > best_val_loss:
                continue
            best_acc = val_acc
            best_val_loss = val_loss

            roi_x, roi_y, roi_w, roi_h = vision_pipeline.ROI
            datetime_str = time.strftime("%Y%m%d_%H%M")
            save_name = (
                f"SC_2ROI_c{len(train_base.classes)}res{IMG_SIZE}"
                f"_batch{BATCH}_lr{LR:.0e}_acc{val_acc:.2f}_e{ep:02d}_{datetime_str}.pt"
            )
            save_path = out / save_name
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_base.classes,
                "img_size": IMG_SIZE,
                "roi": (roi_x, roi_y, roi_w, roi_h),
                "epoch": ep,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_loss": run_loss / seen,
                "batch_size": BATCH,
                "learning_rate": LR
            }, save_path)
            print(f"Saved {save_name}")

    print(f"Training complete. Best val acc={best_acc:.4f}")
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

from pathlib import Path
import time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image

DATA_ROOT = r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data\Classifier"
OUT_DIR = r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\ML\models"

EPOCHS = 5
BATCH = 8
IMG_SIZE = 512  # Set to whatever model expects
LR = 1e-3
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# --- ROI Crop (change only here for different ROI)
roi_x, roi_y, roi_w, roi_h = 528, 63, 1221, 1096  # ROI for one big image, cropped to the borders of the colors of parts


class FixedCrop(object):
    def __call__(self, img):
        return img.crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))


# --- Transforms ---
base_tf = transforms.Compose([
    FixedCrop(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize after ROI crop
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

aug_tf1 = transforms.Compose([
    FixedCrop(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.25, contrast=0.30, saturation=0.12, hue=0.02),
    transforms.RandomAffine(degrees=2, translate=(0.01, 0.01), scale=(0.99, 1.01), fill=(128, 128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

aug_tf2 = transforms.Compose([
    FixedCrop(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.10, hue=0.015),
    transforms.RandomAffine(degrees=1, translate=(0.01, 0.01), scale=(0.995, 1.005), shear=1, fill=(128, 128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_tf = transforms.Compose([
    FixedCrop(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# --- Triple Aug Dataset (unchanged) ---
class TripleAugDataset(Dataset):
    def __init__(self, base_ds, base_tf, aug_tf1, aug_tf2):
        self.base_ds = base_ds
        self.base_tf = base_tf
        self.aug_tf1 = aug_tf1
        self.aug_tf2 = aug_tf2
        self.classes = base_ds.classes

    def __len__(self):
        return len(self.base_ds) * 3

    def __getitem__(self, i):
        j = i // 3
        variant = i % 3
        path, y = self.base_ds.samples[j]
        img = Image.open(path).convert("RGB")
        if variant == 0:
            x = self.base_tf(img)
        elif variant == 1:
            x = self.aug_tf1(img)
        else:
            x = self.aug_tf2(img)
        return x, y


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


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    t0 = time.time()  # Start timer
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    # Datasets
    train_base = datasets.ImageFolder(Path(DATA_ROOT) / "train")
    val_ds = datasets.ImageFolder(Path(DATA_ROOT) / "validation", transform=val_tf)
    train_ds = TripleAugDataset(train_base, base_tf, aug_tf1, aug_tf2)
    print(f"Classes: {train_ds.classes}")
    print(f"Train originals: {len(train_base)}, effective: {len(train_ds)}")
    print(f"Val: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)
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
            roi_tag = f"roi{roi_x}_{roi_y}_{roi_w}_{roi_h}"
            save_name = (
                f"SC_{roi_tag}_res{IMG_SIZE}"
                f"_e{ep:02d}_lr{LR:.0e}_acc{val_acc:.3f}.pt"
            )
            save_path = out / save_name

            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": IMG_SIZE,
                "roi": (roi_x, roi_y, roi_w, roi_h),
                "epoch": ep,
                "val_acc": val_acc
            }, save_path)
            print(f"Saved {save_name}")

    print(f"Training complete in {time.time() - t0:.1f}s. Best val acc={best_acc:.4f}")


if __name__ == "__main__":
    main()

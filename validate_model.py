import torch
import torch.nn as nn
import numpy as np
import os
from vision_pipeline import load_images_npy
from models import Autoencoder

# Configuration
AUTOENCODER_NAME = 'AC_yellow_standard_e60_b4_lr20_res1024.pth'
USE_MANUAL_THRESHOLD = False
MANUAL_THRESHOLD = 0.0008
PATH = 'image_data/validation/processed'


def find_optimal_threshold_f1(losses, labels):
    """Find threshold that maximizes F1-score."""
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}

    for threshold in np.linspace(losses.min(), losses.max(), 100):
        predictions = (losses > threshold).astype(int)

        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }

    return best_threshold, best_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # Detect model type from filename
    use_attention = 'weighted' in AUTOENCODER_NAME.lower()
    use_deep = 'deep' in AUTOENCODER_NAME.lower()

    # Import the correct model class
    if use_deep:
        from models import DeepAutoencoder
        model = DeepAutoencoder(use_attention=use_attention).to(device)
        model_type = "Deep " + ("Weighted" if use_attention else "Standard")
    else:
        from models import Autoencoder
        model = Autoencoder(use_attention=use_attention).to(device)
        model_type = "Weighted" if use_attention else "Standard"

    path = os.path.join('models', AUTOENCODER_NAME)
    model.load_state_dict(torch.load(path, map_location=device))  # Add map_location=device
    model.eval()

    print(f'Model: {model_type} Autoencoder')
    print(f'Threshold: {"Manual" if USE_MANUAL_THRESHOLD else "Optimized (F1)"}\n')

    criterion = nn.MSELoss(reduction='none' if use_attention else 'mean')

    val_images = load_images_npy(PATH)
    val_tensor = torch.from_numpy(val_images).float()
    val_files = sorted([f for f in os.listdir(PATH) if f.endswith('.npy')])

    # Collect losses
    losses = []
    labels = []

    print('Collecting losses...')
    for i, img in enumerate(val_tensor):
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_batch)

            # Handle both standard and weighted models
            if isinstance(output, tuple):
                decoded, attention_mask = output
                pixel_loss = criterion(decoded, img_batch)
                attention_mask_expanded = attention_mask.expand_as(pixel_loss)
                loss = (pixel_loss * attention_mask_expanded).mean().item()
            else:
                loss = criterion(output, img_batch).item()

        losses.append(loss)
        labels.append(1 if 'false' in val_files[i].lower() else 0)

    losses = np.array(losses)
    labels = np.array(labels)

    # Determine threshold
    if USE_MANUAL_THRESHOLD:
        threshold = MANUAL_THRESHOLD
        print(f'Using manual threshold: {threshold:.6f}\n')
    else:
        threshold, metrics = find_optimal_threshold_f1(losses, labels)

        print(f'F1-Score Optimization:')
        print(f'  Threshold: {threshold:.10f}')
        print(f'  Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}')
        print(f'  TP: {metrics["tp"]}, FP: {metrics["fp"]}, TN: {metrics["tn"]}, FN: {metrics["fn"]}\n')

    # Validation results
    print('=' * 100)
    print(f'{"ID":<5} {"Filename":<50} {"Loss":<12} {"Predict":<10} {"True":<10}')
    print('=' * 100)

    for i, (loss, label, filename) in enumerate(zip(losses, labels, val_files)):
        prediction = "DEFECT" if loss > threshold else "OK"
        true_label = "DEFECT" if label == 1 else "OK"
        print(f'{i + 1:<5} {filename:<50} {loss:<12.10f} {prediction:<10} {true_label:<10}')

    print('=' * 100)


if __name__ == '__main__':
    main()

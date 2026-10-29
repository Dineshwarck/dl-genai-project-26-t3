import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.dataset import SurfaceDefectDataset, get_stratified_split
from src.model import SurfaceDefectClassifier
import albumentations as A

def get_classifier_transforms():
    """Augmentations optimized for global defect feature learning."""
    train_tf = A.Compose([
        A.Resize(height=128, width=800),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
    ])
    val_tf = A.Compose([A.Resize(height=128, width=800)])
    return train_tf, val_tf


def train_classifier(
    data_csv='data/train.csv',
    image_dir='data/train',
    epochs=15,
    batch_size=16,          # Larger batch size possible since we only train classifier
    lr=5e-4,
    random_state=42,
    checkpoint_name='best_classifier.pth',
    device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training SurfaceDefectClassifier on: {device}")

    # 1. Resolve Dataset Paths (handles Kaggle, Colab, or local repo root)
    if not os.path.exists(data_csv):
        candidates_csv = [
            '/kaggle/input/competitions/industrial-surface-defect-detection/public/train.csv',
            '/kaggle/input/industrial-surface-defect-detection/public/train.csv',
            '/kaggle/input/industrial-surface-defect-detection/train.csv',
            'data/train.csv',
            '../data/train.csv',
            os.path.join(os.getcwd(), 'data', 'train.csv')
        ]
        for c in candidates_csv:
            if os.path.exists(c):
                data_csv = c
                break

    if not os.path.exists(image_dir):
        candidates_img = [
            '/kaggle/input/competitions/industrial-surface-defect-detection/public/train',
            '/kaggle/input/industrial-surface-defect-detection/public/train',
            '/kaggle/input/industrial-surface-defect-detection/train',
            'data/train',
            '../data/train',
            os.path.join(os.getcwd(), 'data', 'train')
        ]
        for c in candidates_img:
            if os.path.exists(c):
                image_dir = c
                break

    print(f"Using Data CSV: {data_csv}")
    print(f"Using Image Directory: {image_dir}")

    df = pd.read_csv(data_csv)
    train_df, val_df = get_stratified_split(df, val_size=0.20, random_state=random_state)
    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    # 2. DataLoaders
    train_tf, val_tf = get_classifier_transforms()
    train_ds = SurfaceDefectDataset(train_df, image_dir=image_dir, transforms=train_tf)
    val_ds = SurfaceDefectDataset(val_df, image_dir=image_dir, transforms=val_tf)

    num_workers = 2 if (torch.cuda.is_available() and os.name != 'nt') else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 3. Model, Weighted BCE Loss, Optimizer
    model = SurfaceDefectClassifier(in_channels=3, num_classes=4).to(device)

    # Multiple-GPU check
    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print(f"Utilizing {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)

    # Positive class weighting: penalizes missing rare defects (Defect B gets 8x, Defect A gets 3x)
    pos_weights = torch.tensor([3.0, 8.0, 1.2, 1.8], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_val_score = 0.0
    os.makedirs('models', exist_ok=True)
    ckpt_path = os.path.join('models', checkpoint_name)

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images = images.to(device)
            # Convert (B, 4, 128, 800) masks to (B, 4) binary multi-label vectors
            targets = (masks.sum(dim=(2, 3)) > 0).float().to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images = images.to(device)
                targets = (masks.sum(dim=(2, 3)) > 0).float().to(device)

                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    logits = model(images)
                    loss = criterion(logits, targets)

                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.vstack(all_preds)      # (N, 4)
        all_targets = np.vstack(all_targets)  # (N, 4)

        # Binary predictions at standard threshold 0.50
        binary_preds = (all_preds > 0.50).astype(int)

        # Compute Clean Sheet Accuracy: true clean vs predicted clean
        is_truly_clean = (all_targets.sum(axis=1) == 0)
        pred_is_clean = (binary_preds.sum(axis=1) == 0)
        clean_acc = (pred_is_clean[is_truly_clean]).mean() if is_truly_clean.sum() > 0 else 1.0

        # Overall Macro F1 across 4 defect classes
        f1_scores = []
        for c in range(4):
            tp = ((binary_preds[:, c] == 1) & (all_targets[:, c] == 1)).sum()
            fp = ((binary_preds[:, c] == 1) & (all_targets[:, c] == 0)).sum()
            fn = ((binary_preds[:, c] == 0) & (all_targets[:, c] == 1)).sum()
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
            f1_scores.append(f1)
        macro_f1 = np.mean(f1_scores)

        # Balance Score: 60% Macro F1 + 40% Clean Sheet Silence
        val_score = 0.60 * macro_f1 + 0.40 * clean_acc

        print(f"\n--- Epoch {epoch}/{epochs} Results ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Clean Sheet Accuracy: {clean_acc*100:.2f}% | Macro F1: {macro_f1:.4f} | Val Score: {val_score:.4f}")
        print(f"Per-Class F1 -> A: {f1_scores[0]:.3f} | B: {f1_scores[1]:.3f} | C: {f1_scores[2]:.3f} | D: {f1_scores[3]:.3f}")

        if val_score > best_val_score:
            best_val_score = val_score
            save_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({'state_dict': save_dict, 'epoch': epoch, 'val_score': val_score}, ckpt_path)
            print(f"🔥 Saved new best classifier checkpoint to: {ckpt_path} (Score: {best_val_score:.4f})")

    print(f"\n[OK] Training completed! Best Classifier Val Score: {best_val_score:.4f}")
    return model


if __name__ == '__main__':
    train_classifier()

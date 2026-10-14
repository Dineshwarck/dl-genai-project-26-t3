import gc
import os
import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Import modular components
from src.dataset import SurfaceDefectDataset, get_stratified_split
from src.losses import ComboLoss
from src.metrics import compute_competition_metrics
from src.model import ScratchUNet


def get_augmentations():
    train_transform = A.Compose(
        [A.Resize(height=128, width=800), A.HorizontalFlip(p=0.5)]
    )
    val_transform = A.Compose([A.Resize(height=128, width=800)])
    return train_transform, val_transform

def train_model(
    data_csv=r'D:\SurfaceGuard_AI\data\train.csv',
    image_dir=r'D:\SurfaceGuard_AI\data\train',
    epochs=5,
    batch_size=4,
    lr=1e-3,
    sample_limit=None,
):
     # 1. Device Agnostic code
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using device: {device}")

     # 2. Initialize Weights & Biases
     wandb.init(project='24F2004781-t32026', name='scratch-unet-multilabel')

     # 3. Load Data & Stratified 80/20 split
     df = pd.read_csv(data_csv)
     train_df,val_df = get_stratified_split(df,val_size=0.20,random_state=42)

     if sample_limit:
        train_df = train_df.head(sample_limit)
        val_df = val_df.head(sample_limit // 4)
    
     print(
        f"Training images: {len(train_df)} | Validation images: {len(val_df)}"
     )

     # 4 .DataLoaders
     train_tf, val_tf = get_augmentations()
     train_dataset = SurfaceDefectDataset(
         train_df, image_dir = image_dir, transforms=train_tf
     )
     val_dataset = SurfaceDefectDataset(
         val_df, image_dir = image_dir, transforms=val_tf
     )

     train_loader = DataLoader(
        train_dataset, batch_size = batch_size, shuffle =True, num_workers=0
     )
     val_loader = DataLoader(
        val_dataset, batch_size = batch_size, shuffle =False, num_workers=0
     )

     # 5. Model (4 Output Channels), ComboLoss & Optimizer
     model = ScratchUNet(in_channels=3, out_channels=4).to(device)
     criterion = ComboLoss(bce_weight=1.0,dice_weight=1.0,smooth=1.0)
     optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
         optimizer, T_max=epochs, eta_min=1e-6
     )

     os.makedirs('models', exist_ok = True)
     best_score = -1.0

     # 6. Training & Validation Loop
     for epoch in range(epochs):
        # --- Training Phase ----
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images,masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ----
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader,desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                all_preds.append(outputs.cpu())
                all_targets.append(masks.cpu())
        
        avg_val_loss = val_loss / len(val_loader)   


        # --- OFFICIAL METRIC EVALUATION ---
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        metrics = compute_competition_metrics(all_preds_tensor, all_targets_tensor,threshold=0.5)

        # --- LOGGING ---
        print(
            f"Epoch {epoch+1}/{epochs} |"
            f" Train Loss: {avg_train_loss:.4f} |"
            f" Val Loss: {avg_val_loss:.4f} |"
            f" F1-score: {metrics['foreground_dice']:.4f} |"
            f" FP Score: {metrics['false_positive_score']:.4f} |"
            f" Empty Accuracy: {metrics['empty_accuracy']:.4f} |"
            f" Official Score: {metrics['official_score']:.4f}"
        )
        wandb.log({
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'foreground_dice': metrics['foreground_dice'],
                'false_positive_score': metrics['false_positive_score'],
                'empty_accuracy': metrics['empty_accuracy'],
                'official_score': metrics['official_score'],
            })
        
        # --- CHECKPOINT SAVING ---
        if metrics['official_score'] > best_score:
            best_score = metrics['official_score']
            checkpoint_path = "models/best_scratch_unet.pth"
            torch.save(
                model.state_dict(),
                checkpoint_path
            )
            print(f'Saved new best model checkpoint! Score: {best_score:.4f}\n')

            # Stream model directly to W&B cloud storage
            artifact = wandb.Artifact(
                name='scratch-unet-best',
                type='model',
                description=f'Best Scratch UNet at epoch {epoch+1} with score {best_score:.4f}',
                metadata={
                    'epoch': epoch + 1,
                    'val_loss': avg_val_loss,
                    'foreground_dice': metrics['foreground_dice'],
                    'false_positive_score': metrics['false_positive_score'],
                    'empty_accuracy': metrics['empty_accuracy'],
                    'official_score': metrics['official_score'],
                }
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

        # --- STEP SCHEDULER & FREE MEMORY ---
        scheduler.step()
        del all_preds, all_targets, all_preds_tensor, all_targets_tensor
        gc.collect()
    
     print('Training Complete!')
     wandb.finish()
    
if __name__ == '__main__':
    train_model()        
import cv2
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
from src.dataset import SurfaceDefectDataset, get_stratified_split, get_defect_aware_sampler
from src.losses import ComboLoss
from src.metrics import compute_competition_metrics, CLASS_THRESHOLDS
from src.model import ScratchUNet, ResAttentionUNet, PretrainedResNet34UNet
from src.inference import fetch_model_from_wandb, DEFAULT_CLS_GATES


def get_augmentations():
    train_transform = A.Compose([
        A.Resize(height=128, width=800),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.5
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.CoarseDropout(
            max_holes=6,
            max_height=32,
            max_width=64,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=0,
            p=0.3,
        ),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
    ])
    val_transform = A.Compose([A.Resize(height=128, width=800)])
    return train_transform, val_transform

def train_model(
    data_csv=r'D:\SurfaceGuard_AI\data\train.csv',
    image_dir=r'D:\SurfaceGuard_AI\data\train',
    epochs=30,
    batch_size=8,
    lr=3e-4,
    scheduler_type='onecycle',
    resume_from=None,
    model_architecture='scratch_unet',
    deep_supervision=False,
    use_sampler=True,
    balanced_sampler=None,
    use_focal=True,
    focal_gamma=2.0,
    focal_alpha=(0.75, 0.85, 0.50, 0.60),
    pos_weight=(12.0, 25.0, 3.0, 6.0),
    channel_weights=(1.5, 2.5, 1.0, 1.2),
    soft_gating=True,
    cls_gates=None,
    with_attention=False,
    initial_best_score=None,
    use_amp=True,
    random_state=42,
    checkpoint_name='best_scratch_unet.pth',
    artifact_name='scratch-unet-best',
    wandb_run_name='scratch-unet-multilabel',
    sample_limit=None,
    num_workers=None,
    wandb_artifact='scratch-unet-best:latest',
):
     if balanced_sampler is not None:
         use_sampler = balanced_sampler

     # 1. Device Agnostic code
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using device: {device}")

     if num_workers is None:
         num_workers = 0

     # 2. Initialize Weights & Biases
     wandb.init(project='24F2004781-t32026', name=wandb_run_name)

     # 3. Load Data & Stratified 80/20 split
     if not os.path.exists(data_csv):
         for c in [
             '/kaggle/input/competitions/industrial-surface-defect-detection/public/train.csv',
             '/kaggle/input/industrial-surface-defect-detection/public/train.csv',
             '/kaggle/input/industrial-surface-defect-detection/train.csv',
             'data/train.csv',
             '../data/train.csv',
         ]:
             if os.path.exists(c):
                 data_csv = c
                 break

     if not os.path.exists(image_dir):
         for c in [
             '/kaggle/input/competitions/industrial-surface-defect-detection/public/train',
             '/kaggle/input/industrial-surface-defect-detection/public/train',
             '/kaggle/input/industrial-surface-defect-detection/train',
             'data/train',
             '../data/train',
         ]:
             if os.path.exists(c):
                 image_dir = c
                 break

     print(f"Loading data from: {data_csv}")
     print(f"Loading images from: {image_dir}")
     df = pd.read_csv(data_csv)
     train_df, val_df = get_stratified_split(df, val_size=0.20, random_state=random_state)

     if sample_limit:
        train_df = train_df.head(sample_limit)
        val_df = val_df.head(sample_limit // 4)
    
     print(
        f"Training images: {len(train_df)} | Validation images: {len(val_df)}"
     )

     # 4. DataLoaders (with Defect-Aware Weighted Sampling)
     train_tf, val_tf = get_augmentations()
     train_dataset = SurfaceDefectDataset(
         train_df, image_dir=image_dir, transforms=train_tf
     )
     val_dataset = SurfaceDefectDataset(
         val_df, image_dir=image_dir, transforms=val_tf
     )

     if use_sampler and sample_limit is None:
         train_sampler = get_defect_aware_sampler(train_df)
         train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
         )
         print("Using Defect-Aware Balanced Sampler (oversampling defect_b and defect_a)")
     else:
         train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
         )

     val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
     )

     # 5. Model (4 Output Channels), Focal+Dice ComboLoss & Optimizer
     if str(model_architecture).lower() in ('res_attention_unet', 'resattentionunet', 'resattunet'):
         model = ResAttentionUNet(in_channels=3, out_channels=4, deep_supervision=deep_supervision).to(device)
         print(f"Instantiated ResAttentionUNet (scSE Residual Blocks, ASPP Bottleneck, Attention Gates, DeepSupervision={deep_supervision})")
     elif str(model_architecture).lower() in ('pretrained_resnet34', 'resnet34_unet', 'resnet34'):
         model = PretrainedResNet34UNet(in_channels=3, out_channels=4, pretrained=True, with_classifier=True).to(device)
         print("Instantiated PretrainedResNet34UNet (Transfer Learning from ImageNet with Auxiliary Classifier)")
     else:
         model = ScratchUNet(in_channels=3, out_channels=4, dropout_rate=0.3, with_attention=with_attention).to(device)
         print(f"Instantiated ScratchUNet (with_attention={with_attention})")

     # Optional resume checkpoint (local or W&B)
     if resume_from is not None:
         ckpt_path = resume_from
         if not os.path.exists(ckpt_path):
             print(f"[Resume] Checkpoint '{ckpt_path}' not found locally. Attempting W&B fetch...")
             try:
                 ckpt_path = fetch_model_from_wandb(artifact_name=wandb_artifact)
             except Exception as e:
                 print(f"[Resume] Failed to fetch from W&B: {e}. Starting from scratch.")
                 ckpt_path = None

         if ckpt_path and os.path.exists(ckpt_path):
             print(f"[Resume] Loading weights from: {ckpt_path}")
             ckpt = torch.load(ckpt_path, map_location=device)
             state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
             state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
             missing, unexpected = model.load_state_dict(state_dict, strict=False)
             print(f"[Resume] Weights loaded successfully. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
     # Wrap with DataParallel if multiple GPUs available
     if torch.cuda.is_available() and torch.cuda.device_count() > 1:
         print(f"Utilizing {torch.cuda.device_count()} GPUs with DataParallel!")
         model = nn.DataParallel(model) 
     
     criterion = ComboLoss(
         bce_weight=1.0,
         dice_weight=1.0,
         smooth=1.0,
         use_focal=use_focal,
         gamma=focal_gamma,
         alpha=focal_alpha,
         pos_weight=pos_weight,
         channel_weights=channel_weights,
     )
     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
     scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda" and use_amp))

     # Dynamic Learning Rate Scheduler Configuration
     sched_type = str(scheduler_type).lower()
     is_step_per_batch = False
     if sched_type == 'onecycle':
         total_steps = epochs * len(train_loader)
         scheduler = torch.optim.lr_scheduler.OneCycleLR(
             optimizer,
             max_lr=lr,
             total_steps=total_steps,
             pct_start=0.10,
             anneal_strategy='cos',
             div_factor=10.0,
             final_div_factor=100.0,
         )
         is_step_per_batch = True
         print(f"Configured OneCycleLR: {total_steps} steps, pct_start=0.10, max_lr={lr}")
     elif sched_type == 'cosine':
         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
             optimizer, T_max=epochs, eta_min=1e-6
         )
         print(f"Configured CosineAnnealingLR: T_max={epochs}, eta_min=1e-6")
     elif sched_type in ('warm_restarts', 'cosine_restarts'):
         t_0 = max(5, epochs // 3)
         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
             optimizer, T_0=t_0, T_mult=1, eta_min=1e-6
         )
         print(f"Configured CosineAnnealingWarmRestarts: T_0={t_0}, eta_min=1e-6")
     elif sched_type == 'plateau':
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
             optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
         )
         print("Configured ReduceLROnPlateau: mode=max (official_score), factor=0.5, patience=3")
     else:
         raise ValueError(f"Unknown scheduler_type '{scheduler_type}'. Choose 'onecycle', 'cosine', 'warm_restarts', or 'plateau'.")

     # Setup gating thresholds for validation
     if cls_gates is None:
         cls_gates = DEFAULT_CLS_GATES
     cls_gate_tensor = torch.tensor([cls_gates[c] for c in range(4)], device=device).view(1, 4, 1, 1)

     os.makedirs('models', exist_ok=True)
     best_score = initial_best_score if initial_best_score is not None else -1.0

     # 6. Training & Validation Loop
     for epoch in range(epochs):
        # --- Training Phase ----
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda" and use_amp)):
                out = model(images) if getattr(model, 'deep_supervision', False) else model(images, return_cls=True)
                if isinstance(out, tuple) and len(out) == 4:
                    mask_logits, cls_logits, ds3_out, ds2_out = out
                    mask_loss = criterion(mask_logits, masks)
                    ds_loss = 0.2 * criterion(ds3_out, masks) + 0.2 * criterion(ds2_out, masks)
                    cls_targets = (masks.flatten(2).sum(dim=2) > 0).float()
                    cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_logits, cls_targets)
                    total_loss = mask_loss + 0.3 * cls_loss + ds_loss
                else:
                    mask_logits, cls_logits = out
                    mask_loss = criterion(mask_logits, masks)
                    cls_targets = (masks.flatten(2).sum(dim=2) > 0).float()
                    cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_logits, cls_targets)
                    total_loss = mask_loss + 0.3 * cls_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if is_step_per_batch:
                scheduler.step()

            train_loss += total_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase (Memory-Optimized Streaming) ---
        model.eval()
        val_loss = 0.0
        val_intersections = []
        val_pred_sums = []
        val_gt_sums = []

        th_eval_tensor = torch.tensor(
            [CLASS_THRESHOLDS.get(c, 0.30) for c in range(4)], device=device
        ).view(1, 4, 1, 1)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device == "cuda" and use_amp)):
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    images, masks = images.to(device), masks.to(device)
                    mask_logits, cls_logits = model(images, return_cls=True)
                    mask_loss = criterion(mask_logits, masks)

                    cls_targets = (masks.flatten(2).sum(dim=2) > 0).float()
                    cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_logits, cls_targets)
                    total_val_loss = mask_loss + 0.3 * cls_loss
                    val_loss += total_val_loss.item()

                    # Multi-Task Gating: Soft quadratic modulation or hard suppression
                    mask_probs = torch.sigmoid(mask_logits)
                    cls_probs = torch.sigmoid(cls_logits).unsqueeze(-1).unsqueeze(-1)
                    if soft_gating:
                        ratio = torch.clamp(cls_probs / (cls_gate_tensor + 1e-7), max=1.0)
                        gated_probs = mask_probs * (ratio ** 2)
                    else:
                        gated_probs = mask_probs * (cls_probs > cls_gate_tensor).float()

                    # Stream binary metrics per batch without accumulating gigabytes in RAM:
                    preds_bin = (gated_probs > th_eval_tensor).float()
                    preds_flat = preds_bin.flatten(2)  # (B, 4, H*W)
                    masks_flat = masks.flatten(2)      # (B, 4, H*W)

                    inter = (preds_flat * masks_flat).sum(dim=2).cpu()  # (B, 4)
                    p_sum = preds_flat.sum(dim=2).cpu()                 # (B, 4)
                    g_sum = masks_flat.sum(dim=2).cpu()                 # (B, 4)

                    val_intersections.append(inter)
                    val_pred_sums.append(p_sum)
                    val_gt_sums.append(g_sum)

        avg_val_loss = val_loss / len(val_loader)

        # Compute official metrics on compact summary vectors (only ~36 KB total!)
        all_inter = torch.cat(val_intersections, dim=0).flatten()
        all_psum = torch.cat(val_pred_sums, dim=0).flatten()
        all_gsum = torch.cat(val_gt_sums, dim=0).flatten()

        pos_mask = all_gsum > 0
        neg_mask = ~pos_mask

        # 1. Foreground Dice strictly across positive pairs
        foreground_dice = 0.0
        if pos_mask.any():
            dice_pos = (2.0 * all_inter[pos_mask]) / (all_psum[pos_mask] + all_gsum[pos_mask] + 1e-7)
            foreground_dice = dice_pos.mean().item()

        # 2. False Positive Score & Empty Accuracy across clean pairs
        canvas_pixels = 128 * 800
        false_positive_score = 1.0
        empty_accuracy = 1.0
        if neg_mask.any():
            fp_pixels = all_psum[neg_mask]
            fp_frac = fp_pixels / canvas_pixels
            false_positive_score = (1.0 - fp_frac.mean()).item()
            empty_accuracy = (fp_pixels == 0).float().mean().item()

        official_score = (
            (0.70 * foreground_dice)
            + (0.20 * false_positive_score)
            + (0.10 * empty_accuracy)
        )
        metrics = {
            'foreground_dice': foreground_dice,
            'false_positive_score': false_positive_score,
            'empty_accuracy': empty_accuracy,
            'official_score': official_score,
        }

        # --- LOGGING ---
        curr_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1}/{epochs} |"
            f" LR: {curr_lr:.2e} |"
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
            checkpoint_path = os.path.join("models", checkpoint_name)
            
            save_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(
                save_dict,
                checkpoint_path
            )
            print(f'Saved new best model checkpoint! Score: {best_score:.4f}\n')

            # Stream model directly to W&B cloud storage
            artifact = wandb.Artifact(
                name=artifact_name,
                type='model',
                description=f'Best {model.__class__.__name__} at epoch {epoch+1} with score {best_score:.4f}',
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
        if not is_step_per_batch:
            if sched_type == 'plateau':
                scheduler.step(metrics['official_score'])
            else:
                scheduler.step()
        del val_intersections, val_pred_sums, val_gt_sums, all_inter, all_psum, all_gsum
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
     print('Training Complete!')
     wandb.finish()


def fine_tune_model(
    data_csv=r'data/train.csv',
    image_dir=r'data/train',
    checkpoint_path='models/best_scratch_unet.pth',
    wandb_artifact='scratch-unet-best:latest',
    epochs=5,
    lr=4e-5,
    batch_size=8,
    channel_weights=(2.0, 4.0, 1.0, 1.2),  # Boost rare defect_a (2.0) and defect_b (4.0)
    scheduler_type='cosine',
    initial_best_score=0.6990,              # Safety lock: only save if score strictly beats 0.6990
    use_amp=True,
    sample_limit=None,
    num_workers=None,
    **kwargs,
):
    """
    Fine-tunes the best trained ScratchUNet checkpoint with a gentle learning rate,
    rebalanced loss weights for rare defects, and cosine annealing schedule.
    Safe by construction: only overwrites the checkpoint if score strictly beats initial_best_score.
    """
    print("=" * 70)
    print("       STARTING SCRATCH U-NET FINE-TUNING PIPELINE")
    print(f"Target Epochs: {epochs} | LR: {lr} | Schedule: {scheduler_type}")
    print(f"Channel Weights: {channel_weights} (Defect A=2.0, Defect B=4.0)")
    print(f"Baseline Score Threshold: {initial_best_score}")
    print("=" * 70)
    return train_model(
        data_csv=data_csv,
        image_dir=image_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        scheduler_type=scheduler_type,
        resume_from=checkpoint_path,
        channel_weights=channel_weights,
        initial_best_score=initial_best_score,
        wandb_artifact=wandb_artifact,
        wandb_run_name='scratch-unet-finetune',
        use_amp=use_amp,
        sample_limit=sample_limit,
        num_workers=num_workers,
        **kwargs,
    )


if __name__ == '__main__':
    train_model()        
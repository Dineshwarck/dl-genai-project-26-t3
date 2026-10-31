import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
try:
    import wandb
except ImportError:
    wandb = None

from src.dataset import SurfaceDefectDataset, get_stratified_split, get_defect_aware_sampler
from src.losses import ComboLoss
from src.metrics import compute_competition_metrics, post_process_mask, CLASS_THRESHOLDS, MIN_AREAS, MAX_AREAS
from src.model import PretrainedResNet34UNet
from src.inference import DEFAULT_CLS_GATES


def get_pretrained_augmentations():
    """Industrial augmentations optimized for pre-trained visual representations."""
    transforms = [
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
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
    ]

    # Backward/Forward compatible CoarseDropout (Albumentations v1 vs v2)
    try:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(12, 24),
                hole_width_range=(12, 48),
                fill=0,
                p=0.3,
            )
        )
    except TypeError:
        transforms.append(
            A.CoarseDropout(
                max_holes=6,
                max_height=24,
                max_width=48,
                min_holes=1,
                min_height=12,
                min_width=12,
                fill_value=0,
                p=0.3,
            )
        )

    # Backward/Forward compatible GaussNoise
    try:
        transforms.append(A.GaussNoise(std_range=(0.01, 0.04), p=0.2))
    except TypeError:
        transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.2))

    train_tf = A.Compose(transforms)
    val_tf = A.Compose([A.Resize(height=128, width=800)])
    return train_tf, val_tf



def resolve_data_paths(data_csv='data/train.csv', image_dir='data/train'):
    """Resolves data paths across Kaggle, Colab, or local environments."""
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

    return data_csv, image_dir


def train_pretrained_model(
    data_csv='data/train.csv',
    image_dir='data/train',
    epochs=15,
    warmup_epochs=3,
    batch_size=16,
    lr_warmup=5e-4,
    lr_encoder=1e-5,
    lr_decoder=1e-4,
    weight_decay=1e-4,
    val_size=0.20,
    random_state=42,
    use_sampler=True,
    use_amp=True,
    sample_limit=None,
    num_workers=None,
    wandb_project='24F2004781-t32026',
    wandb_run_name='surfaceguard-pretrained-resnet34',
    save_dir='models/pretrained',
    checkpoint_name='best_pretrained_resnet34.pth',
    device=None,
    no_wandb=False,
):
    """
    Trains Model 3: PretrainedResNet34UNet via 2-Phase Progressive Fine-Tuning.

    Phase 1 (Warmup):
      - Pretrained ImageNet encoder is frozen.
      - Trains only the randomly initialized decoder and segmentation head at lr_warmup (5e-4).
      - Prevents catastrophic forgetting of pre-trained weights.

    Phase 2 (Differential Fine-Tuning):
      - Encoder is unfrozen.
      - Trains with discriminative learning rates:
          * Pretrained Backbone: lr_encoder (1e-5) for gentle adaptation to steel textures
          * Custom Decoder & Head: lr_decoder (1e-4) for full task convergence
      - CosineAnnealingLR schedule down to 1e-6.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("      STARTING PRETRAINED RESNET-34 U-NET TRAINING PIPELINE")
    print(f"Device: {device} | Total GPUs Available: {torch.cuda.device_count()}")
    print(f"Total Epochs: {epochs} | Warmup Epochs: {warmup_epochs}")
    print(f"LR Warmup: {lr_warmup} | LR Encoder: {lr_encoder} | LR Decoder: {lr_decoder}")
    print(f"Batch Size: {batch_size} | Mixed Precision (AMP): {use_amp}")
    print("=" * 70)

    # 1. Resolve Data Paths
    data_csv, image_dir = resolve_data_paths(data_csv, image_dir)
    print(f"Resolved Data CSV: {data_csv}")
    print(f"Resolved Image Dir: {image_dir}")

    df = pd.read_csv(data_csv)
    train_df, val_df = get_stratified_split(df, val_size=val_size, random_state=random_state)

    if sample_limit is not None:
        train_df = train_df.head(sample_limit)
        val_df = val_df.head(sample_limit // 4)
        print(f"Sample limit applied: {len(train_df)} train, {len(val_df)} val")
    else:
        print(f"Stratified Split: {len(train_df):,} train images, {len(val_df):,} val images")

    # 2. DataLoaders
    train_tf, val_tf = get_pretrained_augmentations()
    train_ds = SurfaceDefectDataset(train_df, image_dir=image_dir, transforms=train_tf)
    val_ds = SurfaceDefectDataset(val_df, image_dir=image_dir, transforms=val_tf)

    if num_workers is None:
        num_workers = 0

    if use_sampler and sample_limit is None:
        train_sampler = get_defect_aware_sampler(train_df)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=False
        )
        print("Using Defect-Aware Weighted Sampler (oversampling rare Defect A and Defect B)")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # 3. Model Architecture
    model = PretrainedResNet34UNet(
        in_channels=3,
        out_channels=4,
        pretrained=True,
        with_classifier=True
    ).to(device)

    is_parallel = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Utilizing {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
        is_parallel = True

    def get_raw_model():
        return model.module if is_parallel else model

    # 4. Losses: Segmentation ComboLoss + Auxiliary Classification BCE
    seg_criterion = ComboLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        use_focal=True,
        gamma=2.0,
        alpha=(0.75, 0.85, 0.50, 0.60),
        channel_weights=(1.5, 2.5, 1.0, 1.2),
        mask_empty=True
    )
    pos_weights = torch.tensor([3.0, 8.0, 1.2, 1.8], device=device)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda' and use_amp))

    # 5. Initialize Weights & Biases
    use_wandb_logging = False
    if not no_wandb and wandb is not None:
        try:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "model_architecture": "PretrainedResNet34UNet",
                    "backbone": "resnet34 (ImageNet)",
                    "epochs": epochs,
                    "warmup_epochs": warmup_epochs,
                    "batch_size": batch_size,
                    "lr_warmup": lr_warmup,
                    "lr_encoder": lr_encoder,
                    "lr_decoder": lr_decoder,
                    "use_amp": use_amp,
                    "loss_function": "Focal+Dice ComboLoss with Auxiliary BCE",
                    "dataset_size": len(train_df)
                },
                tags=["pretrained", "resnet34", "transfer-learning", "milestone-2"]
            )
            use_wandb_logging = True
            print(f"[W&B] Initialized W&B logging: project='{wandb_project}', run='{wandb_run_name}'")
        except Exception as e:
            print(f"[W&B] Failed to initialize W&B: {e}. Continuing without W&B.")

    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, checkpoint_name)
    best_score = -1.0

    # 6. Training Execution across Phases
    optimizer = None
    scheduler = None
    current_phase = 0

    for epoch in range(epochs):
        # -------------------------------------------------------------
        # Phase Management
        # -------------------------------------------------------------
        if epoch < warmup_epochs and current_phase != 1:
            current_phase = 1
            print(f"\n{'='*65}\n >>> ENTERING PHASE 1: WARMUP (Epochs 1-{warmup_epochs})")
            print("     Encoder is FROZEN. Training only Decoder & Heads at higher LR.")
            print(f"{'='*65}")
            get_raw_model().freeze_encoder()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=lr_warmup, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=1e-5)

        elif epoch >= warmup_epochs and current_phase != 2:
            current_phase = 2
            print(f"\n{'='*65}\n >>> ENTERING PHASE 2: FULL DIFFERENTIAL FINE-TUNING (Epochs {warmup_epochs+1}-{epochs})")
            print("     Encoder is UNFROZEN. Applying Discriminative Learning Rates:")
            print(f"     - Backbone LR: {lr_encoder} | Decoder LR: {lr_decoder}")
            print(f"{'='*65}")
            get_raw_model().unfreeze_encoder()
            param_groups = get_raw_model().get_parameter_groups(
                lr_encoder=lr_encoder, lr_decoder=lr_decoder, weight_decay=weight_decay
            )
            optimizer = torch.optim.AdamW(param_groups)
            fine_tune_epochs = max(1, epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-6)

        # -------------------------------------------------------------
        # Training Step
        # -------------------------------------------------------------
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_cls_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train - Phase {current_phase}]")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            gt_cls = (masks.sum(dim=(2, 3)) > 0).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device == 'cuda' and use_amp)):
                out = model(images, return_cls=True)
                if isinstance(out, (tuple, list)):
                    mask_logits, cls_logits = out
                else:
                    mask_logits = out
                    cls_logits = None

                seg_loss = seg_criterion(mask_logits, masks)
                if cls_logits is not None:
                    cls_loss = cls_criterion(cls_logits, gt_cls)
                    loss = seg_loss + 0.3 * cls_loss
                else:
                    cls_loss = torch.tensor(0.0, device=device)
                    loss = seg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_seg_loss += seg_loss.item() * images.size(0)
            train_cls_loss += cls_loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "seg": f"{seg_loss.item():.4f}"})

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_seg_loss /= len(train_loader.dataset)
        train_cls_loss /= len(train_loader.dataset)

        # -------------------------------------------------------------
        # Validation Step with Soft Quadratic Gating & Metrics
        # -------------------------------------------------------------
        model.eval()
        val_preds_all = []
        val_targets_all = []
        val_loss = 0.0

        cls_gate_tensor = torch.tensor([DEFAULT_CLS_GATES[c] for c in range(4)], device=device).view(1, 4, 1, 1)
        th_tensor = torch.tensor([CLASS_THRESHOLDS[c] for c in range(4)], device=device).view(1, 4, 1, 1)

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                gt_cls = (masks.sum(dim=(2, 3)) > 0).float()

                with torch.amp.autocast('cuda', enabled=(device == 'cuda' and use_amp)):
                    out = model(images, return_cls=True)
                    if isinstance(out, (tuple, list)):
                        mask_logits, cls_logits = out
                    else:
                        mask_logits = out
                        cls_logits = None

                    v_loss = seg_criterion(mask_logits, masks)
                    val_loss += v_loss.item() * images.size(0)

                mask_p = torch.sigmoid(mask_logits)
                if cls_logits is not None:
                    cls_p = torch.sigmoid(cls_logits).unsqueeze(-1).unsqueeze(-1)
                    # Soft Quadratic Gating
                    ratio = torch.clamp(cls_p / (cls_gate_tensor + 1e-7), max=1.0)
                    final_p = mask_p * (ratio ** 2)
                else:
                    final_p = mask_p

                preds_bin = (final_p > th_tensor).cpu().numpy().astype(np.uint8)

                for b in range(preds_bin.shape[0]):
                    clean = post_process_mask(
                        preds_bin[b],
                        min_areas=MIN_AREAS,
                        max_areas=MAX_AREAS,
                        use_closing=True
                    )
                    val_preds_all.append(clean)
                val_targets_all.append(masks.cpu().numpy().astype(np.uint8))

        val_loss /= len(val_loader.dataset)
        val_preds_tensor = torch.tensor(np.array(val_preds_all), dtype=torch.float32)
        val_targets_tensor = torch.tensor(np.concatenate(val_targets_all, axis=0), dtype=torch.float32)

        metrics = compute_competition_metrics(val_preds_tensor, val_targets_tensor, threshold=0.5)
        official_score = metrics['official_score']
        fg_dice = metrics['foreground_dice']
        fp_score = metrics['false_positive_score']
        empty_acc = metrics['empty_accuracy']

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"\n--- Epoch {epoch+1}/{epochs} Summary (Phase {current_phase}) ---\n"
            f"Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Cls: {train_cls_loss:.4f})\n"
            f"Val Loss:   {val_loss:.4f}\n"
            f"Official Score:   {official_score:.5f}\n"
            f"Foreground Dice:  {fg_dice:.5f} (70% weight)\n"
            f"FP Silence Score: {fp_score:.5f} (20% weight)\n"
            f"Empty Accuracy:   {empty_acc:.5f} (10% weight)\n"
            f"Learning Rate:    {current_lr:.2e}"
        )

        # -------------------------------------------------------------
        # W&B Logging
        # -------------------------------------------------------------
        if use_wandb_logging:
            wandb.log({
                "epoch": epoch + 1,
                "phase": current_phase,
                "train/loss": train_loss,
                "train/seg_loss": train_seg_loss,
                "train/cls_loss": train_cls_loss,
                "val/loss": val_loss,
                "val/official_score": official_score,
                "val/foreground_dice": fg_dice,
                "val/false_positive_score": fp_score,
                "val/empty_accuracy": empty_acc,
                "lr": current_lr,
            })

        # -------------------------------------------------------------
        # Save Best Checkpoint
        # -------------------------------------------------------------
        if official_score > best_score:
            best_score = official_score
            state_dict = get_raw_model().state_dict()
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'official_score': official_score,
                'metrics': metrics,
                'model_class': 'PretrainedResNet34UNet',
                'phase': current_phase,
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f" >>> [SAVED BEST MODEL] New Best Score: {best_score:.5f} -> {best_checkpoint_path}")

    print("\n" + "=" * 70)
    print(f" TRAINING COMPLETE | Best Validation Score: {best_score:.5f}")
    print(f" Best Checkpoint Saved: {best_checkpoint_path}")
    print("=" * 70)

    # 7. Upload Artifact to W&B
    if use_wandb_logging:
        try:
            artifact = wandb.Artifact(
                name="pretrained-resnet34-best",
                type="model",
                description="Best Pretrained ResNet-34 U-Net model checkpoint via 2-phase progressive fine-tuning",
                metadata={"best_score": best_score, "epochs": epochs}
            )
            artifact.add_file(best_checkpoint_path)
            wandb.log_artifact(artifact)
            print(f"[W&B] Checkpoint successfully logged as artifact: 'pretrained-resnet34-best:latest'")
        except Exception as e:
            print(f"[W&B] Artifact upload warning: {e}")
        wandb.finish()

    return best_checkpoint_path, best_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Pretrained ResNet-34 U-Net via 2-Phase Fine-Tuning")
    parser.add_argument('--epochs', type=int, default=15, help="Total training epochs")
    parser.add_argument('--warmup-epochs', type=int, default=3, help="Phase 1 frozen encoder warmup epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size per step")
    parser.add_argument('--lr-warmup', type=float, default=5e-4, help="Learning rate for Phase 1 warmup")
    parser.add_argument('--lr-encoder', type=float, default=1e-5, help="Learning rate for backbone in Phase 2")
    parser.add_argument('--lr-decoder', type=float, default=1e-4, help="Learning rate for decoder in Phase 2")
    parser.add_argument('--no-wandb', action='store_true', help="Disable W&B tracking")
    parser.add_argument('--sample-limit', type=int, default=None, help="Limit samples for dry-run testing")
    args = parser.parse_args()

    train_pretrained_model(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        lr_warmup=args.lr_warmup,
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        no_wandb=args.no_wandb,
        sample_limit=args.sample_limit
    )

import os
import sys
import time
import gc
import ctypes
import numpy as np
import pandas as pd
import cv2
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# Apply glibc memory optimizations to prevent heap fragmentation
try:
    ctypes.CDLL('libc.so.6').mallopt(ctypes.c_int(-100), ctypes.c_int(0))
    ctypes.CDLL('libc.so.6').mallopt(ctypes.c_int(-101), ctypes.c_int(1))
except Exception:
    pass

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class CFG:
    SEED = 42
    DATA_CSV = 'data/train.csv'
    IMAGE_DIR = 'data/train'
    SAMPLE_SUB_CSV = 'data/sample_submission.csv'
    TEST_IMAGE_DIR = 'data/test'
    IMG_HEIGHT = 128
    IMG_WIDTH = 800
    IN_CHANNELS = 3
    NUM_CLASSES = 4
    BATCH_SIZE = 16  # Per GPU batch size
    EPOCHS = 40
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    VAL_SIZE = 0.20
    NUM_WORKERS = 0
    DEFECT_CLASSES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']
    CLASS_THRESHOLDS = {0: 0.32, 1: 0.35, 2: 0.30, 3: 0.30}
    MIN_AREAS = {0: 60, 1: 120, 2: 350, 3: 300}
    MAX_AREAS = {0: 25000, 1: 15000, 2: 35000, 3: 35000}
    CLS_GATES = {0: 0.25, 1: 0.20, 2: 0.35, 3: 0.35}
    WANDB_PROJECT = '24F2004781-t32026'
    WANDB_ENTITY = 'dineshwar-developer-indian-institute-of-technology'
    RESUME_IF_CHECKPOINT_EXISTS = True


def resolve_paths():
    csv_candidates = [
        '/kaggle/input/competitions/industrial-surface-defect-detection/public/train.csv',
        '/kaggle/input/industrial-surface-defect-detection/public/train.csv',
        '/kaggle/input/industrial-surface-defect-detection/train.csv',
        'data/train.csv'
    ]
    for cand in csv_candidates:
        if os.path.exists(cand):
            CFG.DATA_CSV = cand
            break

    img_candidates = [
        '/kaggle/input/competitions/industrial-surface-defect-detection/public/train',
        '/kaggle/input/industrial-surface-defect-detection/public/train',
        '/kaggle/input/industrial-surface-defect-detection/train',
        'data/train'
    ]
    for cand in img_candidates:
        if os.path.exists(cand):
            CFG.IMAGE_DIR = cand
            break


def rle_to_mask(rle_string, height=128, width=800):
    if not isinstance(rle_string, str) or not rle_string.strip():
        return np.zeros((height, width), dtype=np.uint8)
    runs = np.fromstring(rle_string, dtype=np.int32, sep=' ')
    starts = runs[::2] - 1
    lengths = runs[1::2]
    mask_flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask_flat[start:start + length] = 1
    return mask_flat.reshape((height, width), order='F')


class SurfaceDefectDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        if 'defect_type' in df.columns:
            pivoted = df.pivot(index='image_id', columns='defect_type', values='mask_rle').reset_index()
        else:
            pivoted = df.copy().reset_index(drop=True)
        self.image_ids = pivoted['image_id'].astype(str).values
        self.rles = {
            c: np.array([str(x) if (pd.notna(x) and str(x).strip() != '') else '' for x in pivoted[c].values], dtype=object)
            for c in CFG.DEFECT_CLASSES
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg" if not image_id.endswith('.jpg') else image_id)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        mask = np.zeros((h, w, 4), dtype=np.uint8)
        for c, defect in enumerate(CFG.DEFECT_CLASSES):
            rle_str = self.rles[defect][idx]
            if rle_str:
                mask[:, :, c] = rle_to_mask(rle_str, height=h, width=w)
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image_np = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        mask_np = np.ascontiguousarray(mask.transpose(2, 0, 1), dtype=np.float32)
        return torch.from_numpy(image_np), torch.from_numpy(mask_np)


def get_stratified_split(df, val_size=0.20, random_state=42):
    if 'defect_type' in df.columns:
        pivoted = df.pivot(index='image_id', columns='defect_type', values='mask_rle').reset_index()
    else:
        pivoted = df.copy()

    def get_signature(row):
        return tuple(int(pd.notna(row[c]) and str(row[c]).strip() != '') for c in CFG.DEFECT_CLASSES)

    pivoted['signature'] = pivoted.apply(get_signature, axis=1)
    from sklearn.model_selection import StratifiedShuffleSplit
    strat_key = pivoted['signature'].apply(lambda sig: 'empty' if sum(sig) == 0 else f"comb_{''.join(map(str, sig))}")
    counts = strat_key.value_counts()
    rare_classes = counts[counts < 2].index
    strat_key_binned = strat_key.apply(lambda k: 'rare_defect' if k in rare_classes else k)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(pivoted, strat_key_binned))
    train_df = pivoted.iloc[train_idx].drop(columns=['signature']).reset_index(drop=True)
    val_df = pivoted.iloc[val_idx].drop(columns=['signature']).reset_index(drop=True)
    return train_df, val_df


def get_defect_aware_weights(df, clean_weight=0.2, a_weight=4.0, b_weight=12.0, c_weight=1.2, d_weight=1.5):
    weights = np.full(len(df), clean_weight, dtype=np.float32)
    for idx in range(len(df)):
        row = df.iloc[idx]
        w = clean_weight
        if pd.notna(row.get('defect_c')) and str(row.get('defect_c')).strip() != '':
            w = max(w, c_weight)
        if pd.notna(row.get('defect_d')) and str(row.get('defect_d')).strip() != '':
            w = max(w, d_weight)
        if pd.notna(row.get('defect_a')) and str(row.get('defect_a')).strip() != '':
            w = max(w, a_weight)
        if pd.notna(row.get('defect_b')) and str(row.get('defect_b')).strip() != '':
            w = max(w, b_weight)
        weights[idx] = w
    return torch.tensor(weights, dtype=torch.double)


# Architecture Definitions
class scSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // reduction, 4), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // reduction, 4), in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ResBlockWithscSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = scSEBlock(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.scse(out)
        return self.relu(out + res)


class ASPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        out4 = F.interpolate(self.b4(x), size=(h, w), mode='bilinear', align_corners=True)
        return self.project(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), out4], dim=1))


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResAttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.enc1 = ResBlockWithscSE(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ResBlockWithscSE(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ResBlockWithscSE(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = ResBlockWithscSE(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = ASPPBottleneck(512, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=256, F_l=512, F_int=256)
        self.dec4 = ResBlockWithscSE(512 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=256, F_int=128)
        self.dec3 = ResBlockWithscSE(256 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=64, F_l=128, F_int=64)
        self.dec2 = ResBlockWithscSE(128 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=32, F_l=64, F_int=32)
        self.dec1 = ResBlockWithscSE(64 + 32, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, out_channels)
        )

    def forward(self, x, return_cls=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        x4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        mask_logits = self.final_conv(d1)
        if return_cls:
            cls_logits = self.cls_head(b)
            return mask_logits, cls_logits
        return mask_logits


class ComboLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, use_focal=True, gamma=2.0, alpha=(0.75, 0.85, 0.50, 0.60), channel_weights=(1.5, 2.5, 1.0, 1.2)):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = (alpha_t * ((1.0 - p_t) ** self.gamma) * bce * self.channel_weights).mean()

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = (1.0 - (2.0 * intersection + 1e-7) / (union + 1e-7)).mean()
        return self.bce_weight * focal_loss + self.dice_weight * dice_loss


def train_ddp():
    # 1. Distributed Setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_distributed:
        torch.cuda.set_device(local_rank)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    resolve_paths()

    if rank == 0:
        print("=" * 70)
        print(f"  STARTING 2-GPU DISTRIBUTEDDATAPARALLEL (DDP) TRAINING (40 EPOCHS)")
        print(f"  World Size: {world_size} GPUs | Master Rank: {rank} | Per-GPU Batch: {CFG.BATCH_SIZE}")
        print(f"  Effective Global Batch Size: {CFG.BATCH_SIZE * world_size}")
        print("=" * 70)

    # 2. Dataset & Transforms
    train_tf = A.Compose([
        A.Resize(height=CFG.IMG_HEIGHT, width=CFG.IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.4),
    ])
    val_tf = A.Compose([A.Resize(height=CFG.IMG_HEIGHT, width=CFG.IMG_WIDTH)])

    df = pd.read_csv(CFG.DATA_CSV)
    train_df, val_df = get_stratified_split(df, val_size=CFG.VAL_SIZE, random_state=CFG.SEED)

    train_dataset = SurfaceDefectDataset(train_df, image_dir=CFG.IMAGE_DIR, transforms=train_tf)
    val_dataset = SurfaceDefectDataset(val_df, image_dir=CFG.IMAGE_DIR, transforms=val_tf)

    weights_tensor = get_defect_aware_weights(train_df)

    # 3. Model, Loss, Optimizer, DDP Wrapper
    model = ResAttentionUNet(in_channels=3, out_channels=4).to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    criterion = ComboLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    
    # Calculate global steps based on effective batch size
    batches_per_epoch = len(train_df) // (CFG.BATCH_SIZE * world_size)
    total_steps = CFG.EPOCHS * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG.LR, total_steps=total_steps,
        pct_start=0.10, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    cls_gate_tensor = torch.tensor([CFG.CLS_GATES[c] for c in range(4)], device=device).view(1, 4, 1, 1)
    th_tensor = torch.tensor([CFG.CLASS_THRESHOLDS[c] for c in range(4)], device=device).view(1, 4, 1, 1)

    os.makedirs('models', exist_ok=True)
    best_score = -1.0
    start_epoch = 1

    # Resume support
    ckpt_path = 'models/best_resattunet.pth'
    if CFG.RESUME_IF_CHECKPOINT_EXISTS and os.path.exists(ckpt_path):
        if rank == 0:
            print(f"  >> [Rank 0] Resuming weights from existing checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt['state_dict'] if (isinstance(ckpt, dict) and 'state_dict' in ckpt) else ckpt
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            target_model = model.module if hasattr(model, 'module') else model
            target_model.load_state_dict(state_dict, strict=False)
            if isinstance(ckpt, dict):
                if 'epoch' in ckpt:
                    start_epoch = int(ckpt['epoch']) + 1
                if 'official_score' in ckpt:
                    best_score = float(ckpt['official_score'])
            if rank == 0:
                print(f"  >> Resumed weights successfully! Starting from Epoch {start_epoch:02d} (Best Score: {best_score:.4f})")
        except Exception as e:
            if rank == 0:
                print(f"  >> Notice: Could not resume from checkpoint ({e}). Starting fresh.")

    # Only Rank 0 initializes W&B
    run = None
    if rank == 0 and wandb is not None:
        try:
            run = wandb.init(
                project=CFG.WANDB_PROJECT,
                entity=CFG.WANDB_ENTITY,
                name=f"surfaceguard-resattunet-ddp-{int(time.time())}",
                config={
                    'epochs': CFG.EPOCHS,
                    'batch_size_per_gpu': CFG.BATCH_SIZE,
                    'world_size': world_size,
                    'effective_batch_size': CFG.BATCH_SIZE * world_size,
                    'lr': CFG.LR,
                    'architecture': 'ResAttentionUNet'
                }
            )
            print("  >> [Rank 0] W&B run initialized successfully.")
        except Exception as e:
            print(f"  >> [Rank 0] W&B initialization notice: {e}")

    # Validation DataLoader (Partitioned deterministically across GPUs)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, sampler=val_sampler, shuffle=(val_sampler is None), num_workers=0, pin_memory=False)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'foreground_dice': [], 'false_positive_score': [], 'official_score': []}
    run_start = time.time()

    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        # Distributed defect-aware sampling: independent seed per rank ensures disjoint batches
        g = torch.Generator()
        g.manual_seed(CFG.SEED + rank + epoch * 100)
        samples_per_gpu = len(train_df) // world_size
        sampled_indices = torch.multinomial(weights_tensor, num_samples=samples_per_gpu, replacement=True, generator=g)
        train_sampler = torch.utils.data.SubsetRandomSampler(sampled_indices)
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=False)

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{CFG.EPOCHS:02d} [Rank {rank} Train]", disable=(rank != 0))
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                mask_logits, cls_logits = model(images, return_cls=True)
                mask_loss = criterion(mask_logits, masks)
                cls_targets = (masks.flatten(2).sum(dim=2) > 0).float()
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)
                total_loss = mask_loss + 0.3 * cls_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += total_loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            if rank == 0:
                train_pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'lr': f"{current_lr:.2e}"})
                loss_val = total_loss.item()
                if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
                    if run is not None:
                        global_step = (epoch - 1) * len(train_loader) + (batch_idx + 1)
                        run.log({
                            'resattunet/batch_loss': loss_val,
                            'resattunet/learning_rate': current_lr,
                            'resattunet/step': global_step
                        }, commit=False)

            del images, masks, mask_logits, cls_logits, mask_loss, cls_targets, cls_loss, total_loss

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        dice_sum = 0.0
        n_pos = 0.0
        fp_sum = 0.0
        empty_correct = 0.0
        n_neg = 0.0

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    mask_logits, cls_logits = model(images, return_cls=True)
                    v_loss = criterion(mask_logits, masks)
                    cls_targets = (masks.flatten(2).sum(dim=2) > 0).float()
                    cls_loss_v = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)
                    val_loss += (v_loss + 0.3 * cls_loss_v).item()

                    mask_probs = torch.sigmoid(mask_logits)
                    cls_probs = torch.sigmoid(cls_logits).unsqueeze(-1).unsqueeze(-1)
                    ratio = torch.clamp(cls_probs / (cls_gate_tensor + 1e-7), max=1.0)
                    gated = mask_probs * (ratio ** 2)
                    preds_bin = (gated > th_tensor).float()

                    pf = preds_bin.flatten(2)
                    mf = masks.flatten(2)
                    inter = (pf * mf).sum(dim=2)
                    pred_s = pf.sum(dim=2)
                    gt_s = mf.sum(dim=2)

                    pos_m = gt_s > 0
                    neg_m = ~pos_m

                    if pos_m.any():
                        b_dice = (2.0 * inter[pos_m]) / (pred_s[pos_m] + gt_s[pos_m] + 1e-7)
                        dice_sum += b_dice.sum().item()
                        n_pos += pos_m.sum().item()

                    if neg_m.any():
                        b_fp = pred_s[neg_m] / (128.0 * 800.0)
                        fp_sum += b_fp.sum().item()
                        empty_correct += (pred_s[neg_m] == 0).sum().item()
                        n_neg += neg_m.sum().item()

                    del images, masks, mask_logits, cls_logits, v_loss, cls_targets, cls_loss_v
                    del mask_probs, cls_probs, ratio, gated, preds_bin, pf, mf, inter, pred_s, gt_s, pos_m, neg_m

        # Aggregate scalar metrics across all DDP GPUs via NCCL all_reduce
        metrics_tensor = torch.tensor(
            [val_loss, dice_sum, n_pos, fp_sum, empty_correct, n_neg, len(val_loader)],
            device=device, dtype=torch.float64
        )
        if is_distributed:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = metrics_tensor[0].item() / metrics_tensor[6].item() if metrics_tensor[6].item() > 0 else 0.0
        fg_dice = (metrics_tensor[1].item() / metrics_tensor[2].item()) if metrics_tensor[2].item() > 0 else 0.0
        fp_score = 1.0 - (metrics_tensor[3].item() / metrics_tensor[5].item()) if metrics_tensor[5].item() > 0 else 1.0
        empty_acc = (metrics_tensor[4].item() / metrics_tensor[5].item()) if metrics_tensor[5].item() > 0 else 1.0
        official = 0.70 * fg_dice + 0.20 * fp_score + 0.10 * empty_acc

        epoch_duration = time.time() - epoch_start
        elapsed_min = (time.time() - run_start) / 60.0
        avg_epoch_sec = (time.time() - run_start) / (epoch - start_epoch + 1)
        eta_min = ((CFG.EPOCHS - epoch) * avg_epoch_sec) / 60.0

        if rank == 0:
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['foreground_dice'].append(fg_dice)
            history['false_positive_score'].append(fp_score)
            history['official_score'].append(official)

            print(f"Epoch {epoch:02d}/{CFG.EPOCHS:02d} | Time: {epoch_duration:.1f}s (Elapsed: {elapsed_min:.1f}m | ETA: {eta_min:.1f}m) | "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"Dice: {fg_dice:.4f} | FP: {fp_score:.4f} | Score: {official:.4f}")

            if run is not None:
                run.log({
                    'resattunet/epoch': epoch,
                    'resattunet/train_loss': avg_train_loss,
                    'resattunet/val_loss': avg_val_loss,
                    'resattunet/foreground_dice': fg_dice,
                    'resattunet/false_positive_score': fp_score,
                    'resattunet/official_score': official,
                    'resattunet/epoch_duration_sec': epoch_duration,
                    'resattunet/elapsed_minutes': elapsed_min,
                    'resattunet/eta_minutes': eta_min,
                })

            if official > best_score:
                best_score = official
                raw_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                save_payload = {
                    'state_dict': raw_dict,
                    'epoch': epoch,
                    'official_score': float(best_score),
                    'history': history,
                    'metrics': {
                        'foreground_dice': fg_dice,
                        'false_positive_score': fp_score,
                        'official_score': official,
                    }
                }
                torch.save(save_payload, 'models/best_resattunet.pth')
                del raw_dict, save_payload
                print(f"  >> [Rank 0] Saved new best model! Score: {best_score:.4f}")

                # Immediate in-loop W&B artifact upload
                if run is not None:
                    try:
                        artifact = wandb.Artifact(
                            name='resattunet-best',
                            type='model',
                            description=f'Best ResAttentionUNet checkpoint at Epoch {epoch:02d} (Score: {best_score:.4f})',
                            metadata={'official_score': float(best_score), 'epoch': epoch, 'epochs': CFG.EPOCHS}
                        )
                        artifact.add_file('models/best_resattunet.pth')
                        run.log_artifact(artifact)
                        print(f"  >> [Rank 0 Artifact] Uploaded 'resattunet-best' to W&B (Epoch {epoch:02d}, Score: {best_score:.4f})")
                    except Exception as e:
                        print(f"  >> [Rank 0 Artifact] Upload notice: {e}")

        del train_pbar, train_loader, train_sampler
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        print(f"\n[Rank 0] Training complete. Best official score: {best_score:.4f}")


if __name__ == '__main__':
    train_ddp()

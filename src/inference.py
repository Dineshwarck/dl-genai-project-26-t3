import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.model import ScratchUNet, ResAttentionUNet
from src.metrics import post_process_mask, CLASS_THRESHOLDS, MIN_AREAS, MAX_AREAS
from src.utils import mask_to_rle
import wandb

def fetch_model_from_wandb(
    project='24F2004781-t32026',
    artifact_name='scratch-unet-best:latest',
    target_dir='models'
):
    """
    Downloads the best model checkpoint from Weights & Biases Artifacts.
    Returns the path to the downloaded .pth file.
    """
    print(f"[W&B] Fetching artifact '{artifact_name}' from W&B project '{project}'...")
    api = wandb.Api()
    
    # Resolves and downloads the artifact to target_dir
    artifact = api.artifact(f"{project}/{artifact_name}", type='model')
    download_dir = artifact.download(root=target_dir)
    
    pth_files = [f for f in os.listdir(download_dir) if f.endswith('.pth')]
    if pth_files:
        checkpoint_path = os.path.join(download_dir, pth_files[0])
    else:
        checkpoint_path = os.path.join(download_dir, 'best_scratch_unet.pth')
    print(f"[OK] Successfully downloaded best model to: {checkpoint_path}")
    return checkpoint_path




DEFECT_CLASSES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

class TestDefectDataset(Dataset):
    """Loads test images without ground-truth masks."""
    def __init__(self, image_ids, image_dir):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (800, 128))  # (W, H) = (800, 128)
        
        # (H, W, 3) -> (3, H, W) normalized to [0, 1]
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image_tensor, img_id

DEFAULT_CLS_GATES = {
    0: 0.25,  # defect_a: lower gate to preserve small inclusions/pitting
    1: 0.20,  # defect_b: sensitive gate for rare edge cracks/tears
    2: 0.35,  # defect_c: canonical gate to balance foreground dice and clean steel (crazing/scaling)
    3: 0.35,  # defect_d: canonical gate for surface patches
}


def load_single_model(checkpoint_path, device):
    """Loads a ScratchUNet or ResAttentionUNet model checkpoint and auto-detects architecture."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    is_res_att = any('scse' in k or 'b0' in k for k in state_dict.keys())
    if is_res_att:
        has_ds = any(k.startswith('ds') for k in state_dict.keys())
        model = ResAttentionUNet(in_channels=3, out_channels=4, deep_supervision=has_ds).to(device)
    else:
        has_cls = any(k.startswith('cls_head') for k in state_dict.keys())
        has_att = any(k.startswith('att') for k in state_dict.keys())
        model = ScratchUNet(in_channels=3, out_channels=4, with_classifier=has_cls, with_attention=has_att).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded weights from {checkpoint_path} (Model: {model.__class__.__name__})")
    return model


def predict_batch_probs(models, images, use_tta=True, cls_gate_tensor=None, soft_gating=True):
    """
    Computes ensemble-averaged, classification-gated probability maps for a batch of images.
    Supports single or multiple models, 4-way TTA (Orig, H-Flip, V-Flip, Both), and soft quadratic gating.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    model_probs = []

    if use_tta:
        img_orig = images
        img_h = torch.flip(images, dims=[3])
        img_v = torch.flip(images, dims=[2])
        img_hv = torch.flip(images, dims=[2, 3])

    for model in models:
        with_cls = getattr(model, 'with_classifier', False)
        if use_tta:
            out1 = model(img_orig, return_cls=True) if with_cls else model(img_orig)
            out2 = model(img_h, return_cls=True) if with_cls else model(img_h)
            out3 = model(img_v, return_cls=True) if with_cls else model(img_v)
            out4 = model(img_hv, return_cls=True) if with_cls else model(img_hv)

            if isinstance(out1, (tuple, list)):
                m1, c1 = out1
                m2, c2 = out2
                m3, c3 = out3
                m4, c4 = out4

                m1_p = torch.sigmoid(m1)
                m2_p = torch.flip(torch.sigmoid(m2), dims=[3])
                m3_p = torch.flip(torch.sigmoid(m3), dims=[2])
                m4_p = torch.flip(torch.sigmoid(m4), dims=[2, 3])
                # 4-way TTA with equal weighting (0.25 each) for maximum variance reduction:
                mask_p = 0.25 * (m1_p + m2_p + m3_p + m4_p)

                c1_p = torch.sigmoid(c1)
                c2_p = torch.sigmoid(c2)
                c3_p = torch.sigmoid(c3)
                c4_p = torch.sigmoid(c4)
                cls_p = 0.25 * (c1_p + c2_p + c3_p + c4_p)
                cls_p_4d = cls_p.unsqueeze(-1).unsqueeze(-1)

                if cls_gate_tensor is not None:
                    if soft_gating:
                        # Smooth quadratic attenuation: 1.0 when confident, (p / gate)^2 when below
                        ratio = torch.clamp(cls_p_4d / (cls_gate_tensor + 1e-7), max=1.0)
                        gate = ratio ** 2
                    else:
                        gate = (cls_p_4d > cls_gate_tensor).float()
                    p = mask_p * gate
                else:
                    p = mask_p
            else:
                m1_p = torch.sigmoid(out1)
                m2_p = torch.flip(torch.sigmoid(out2), dims=[3])
                m3_p = torch.flip(torch.sigmoid(out3), dims=[2])
                m4_p = torch.flip(torch.sigmoid(out4), dims=[2, 3])
                p = 0.25 * (m1_p + m2_p + m3_p + m4_p)
        else:
            out = model(images, return_cls=True) if with_cls else model(images)
            if isinstance(out, (tuple, list)):
                m, c = out
                mask_p = torch.sigmoid(m)
                cls_p = torch.sigmoid(c)
                cls_p_4d = cls_p.unsqueeze(-1).unsqueeze(-1)

                if cls_gate_tensor is not None:
                    if soft_gating:
                        ratio = torch.clamp(cls_p_4d / (cls_gate_tensor + 1e-7), max=1.0)
                        gate = ratio ** 2
                    else:
                        gate = (cls_p_4d > cls_gate_tensor).float()
                    p = mask_p * gate
                else:
                    p = mask_p
            else:
                p = torch.sigmoid(out)

        model_probs.append(p)

    return torch.stack(model_probs, dim=0).mean(dim=0)


def generate_submission(
    sample_sub_csv='data/sample_submission.csv',
    test_image_dir='data/test',
    checkpoint_path=None,
    wandb_artifact='scratch-unet-best:latest',
    output_csv='submission.csv',
    threshold=CLASS_THRESHOLDS,
    min_areas=MIN_AREAS,
    max_areas=MAX_AREAS,
    cls_gates=DEFAULT_CLS_GATES,
    soft_gating=True,
    ensemble_paths=None,
    filter_geometry=False,
    use_closing=True,
    use_tta=True,
    batch_size=8,
    sample_limit=None,
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on device: {device}")

    # 1. Resolve Model Checkpoints (Auto-fetch from W&B if local file does not exist)
    def _resolve(p):
        if p and os.path.exists(p):
            return p
        print(f"Local checkpoint '{p}' not found on disk. Fetching '{wandb_artifact}' from W&B...")
        return fetch_model_from_wandb(artifact_name=wandb_artifact)

    model_paths = []
    if checkpoint_path:
        if isinstance(checkpoint_path, (list, tuple)):
            model_paths.extend([_resolve(p) for p in checkpoint_path])
        else:
            model_paths.append(_resolve(checkpoint_path))
    elif wandb_artifact:
        model_paths.append(fetch_model_from_wandb(artifact_name=wandb_artifact))
    else:
        model_paths.append(_resolve('models/best_scratch_unet.pth'))

    if ensemble_paths:
        if isinstance(ensemble_paths, (list, tuple)):
            model_paths.extend(ensemble_paths)
        else:
            model_paths.append(ensemble_paths)

    # 2. Resolve Test Dataset Paths
    if not os.path.exists(sample_sub_csv):
        for c in [
            '/kaggle/input/competitions/industrial-surface-defect-detection/public/sample_submission.csv',
            '/kaggle/input/industrial-surface-defect-detection/public/sample_submission.csv',
            '/kaggle/input/industrial-surface-defect-detection/sample_submission.csv',
            'data/sample_submission.csv',
            '../data/sample_submission.csv',
        ]:
            if os.path.exists(c):
                sample_sub_csv = c
                break

    if not os.path.exists(test_image_dir):
        for c in [
            '/kaggle/input/competitions/industrial-surface-defect-detection/public/test',
            '/kaggle/input/industrial-surface-defect-detection/public/test',
            '/kaggle/input/industrial-surface-defect-detection/test',
            'data/test',
            '../data/test',
        ]:
            if os.path.exists(c):
                test_image_dir = c
                break

    print(f"Using Sample Submission CSV: {sample_sub_csv}")
    print(f"Using Test Image Directory: {test_image_dir}")

    # Load Submission Template
    sub_df = pd.read_csv(sample_sub_csv)
    unique_image_ids = sub_df['image_id'].unique()
    print(f"Total test images to predict: {len(unique_image_ids)}")

    if sample_limit is not None:
        unique_image_ids = unique_image_ids[:sample_limit]
        sub_df = sub_df[sub_df['image_id'].isin(unique_image_ids)].copy()
        print(f"Limited to {sample_limit} images for fast testing")

    # 3. Setup Test DataLoader
    num_workers = 0
    test_dataset = TestDefectDataset(unique_image_ids, test_image_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 4. Load Models
    models = [load_single_model(p, device) for p in model_paths]

    # Setup per-channel threshold tensor
    if isinstance(threshold, (int, float)):
        th_tensor = torch.full((1, 4, 1, 1), float(threshold), device=device)
    elif isinstance(threshold, dict):
        th_tensor = torch.tensor([threshold.get(i, 0.5) for i in range(4)], device=device, dtype=torch.float32).view(1, 4, 1, 1)
    elif isinstance(threshold, (list, tuple)):
        th_tensor = torch.tensor(threshold, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    else:
        th_tensor = torch.full((1, 4, 1, 1), 0.5, device=device)

    # Setup per-channel classifier gate tensor
    if isinstance(cls_gates, (int, float)):
        cls_gate_tensor = torch.full((1, 4, 1, 1), float(cls_gates), device=device)
    elif isinstance(cls_gates, dict):
        cls_gate_tensor = torch.tensor([cls_gates.get(i, 0.35) for i in range(4)], device=device, dtype=torch.float32).view(1, 4, 1, 1)
    elif isinstance(cls_gates, (list, tuple)):
        cls_gate_tensor = torch.tensor(cls_gates, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    else:
        cls_gate_tensor = None

    # 5. Predict & Post-process
    predictions = {}  # Map (image_id, defect_type) -> rle_string

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc=f"Generating Predictions{' (with TTA)' if use_tta else ''}"):
            images = images.to(device)
            probs = predict_batch_probs(
                models,
                images,
                use_tta=use_tta,
                cls_gate_tensor=cls_gate_tensor,
                soft_gating=soft_gating,
            )

            preds_bin = (probs > th_tensor).cpu().numpy().astype(np.uint8)  # (B, 4, 128, 800)

            for i, img_id in enumerate(img_ids):
                # Clean small noise blobs and enforce morphology/area bounds
                cleaned_mask_4ch = post_process_mask(
                    preds_bin[i],
                    min_areas=min_areas,
                    max_areas=max_areas,
                    filter_geometry=filter_geometry,
                    use_closing=use_closing,
                )

                for c, defect_name in enumerate(DEFECT_CLASSES):
                    channel_mask = cleaned_mask_4ch[c]
                    if channel_mask.sum() > 0:
                        rle_str = mask_to_rle(channel_mask)
                    else:
                        rle_str = " "
                    predictions[(img_id, defect_name)] = rle_str

    # 6. Populate Template
    sub_df['mask_rle'] = sub_df.apply(
        lambda row: predictions.get((row['image_id'], row['defect_type']), " "),
        axis=1
    )

    # 7. Save & Summary
    sub_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Submission successfully saved to: {output_csv}")

    # Quick Sanity Check
    non_empty_count = (sub_df['mask_rle'].str.strip() != "").sum()
    print(f"Submission Shape: {sub_df.shape} (Expected: (12000, 4))")
    print(f"Detected Defect Rows: {non_empty_count} / {len(sub_df)} ({non_empty_count/len(sub_df)*100:.2f}%)")
    print(f"Clean Rows: {len(sub_df) - non_empty_count}")


if __name__ == '__main__':
    generate_submission()

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.model import ScratchUNet
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
    0: 0.25,  # defect_a: lower gate to preserve thin 1-pixel scratches
    1: 0.20,  # defect_b: sensitive gate for rare edge cracks
    2: 0.35,  # defect_c: standard gate to suppress background false alarms
    3: 0.35,  # defect_d: standard gate
}


def load_single_model(checkpoint_path, device):
    """Loads a ScratchUNet model checkpoint and auto-detects classifier head."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    has_cls = any(k.startswith('cls_head') for k in state_dict.keys())
    model = ScratchUNet(in_channels=3, out_channels=4, with_classifier=has_cls).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded weights from {checkpoint_path} (with_classifier={has_cls})")
    return model


def predict_batch_probs(models, images, use_tta=True, cls_gate_tensor=None, soft_gating=True):
    """
    Computes ensemble-averaged, classification-gated probability maps for a batch of images.
    Supports single or multiple models, horizontal TTA, and smooth soft quadratic gating.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    model_probs = []
    images_hflip = torch.flip(images, dims=[3]) if use_tta else None

    for model in models:
        with_cls = getattr(model, 'with_classifier', False)
        if use_tta:
            out1 = model(images, return_cls=True) if with_cls else model(images)
            out2 = model(images_hflip, return_cls=True) if with_cls else model(images_hflip)

            if isinstance(out1, (tuple, list)):
                m1, c1 = out1
                m2, c2 = out2
                mask_p = 0.5 * (torch.sigmoid(m1) + torch.flip(torch.sigmoid(m2), dims=[3]))
                cls_p = 0.5 * (torch.sigmoid(c1) + torch.sigmoid(c2))
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
                p = 0.5 * (torch.sigmoid(out1) + torch.flip(torch.sigmoid(out2), dims=[3]))
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
    filter_geometry=True,
    use_tta=True,
    batch_size=8,
    sample_limit=None,
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on device: {device}")

    # 1. Resolve Model Checkpoints (Supports single or ensemble list)
    model_paths = []
    if checkpoint_path:
        if isinstance(checkpoint_path, (list, tuple)):
            model_paths.extend(checkpoint_path)
        else:
            model_paths.append(checkpoint_path)
    elif wandb_artifact:
        model_paths.append(fetch_model_from_wandb(artifact_name=wandb_artifact))
    else:
        model_paths.append('models/best_scratch_unet.pth')

    if ensemble_paths:
        if isinstance(ensemble_paths, (list, tuple)):
            model_paths.extend(ensemble_paths)
        else:
            model_paths.append(ensemble_paths)

    # 2. Load Submission Template
    sub_df = pd.read_csv(sample_sub_csv)
    unique_image_ids = sub_df['image_id'].unique()
    print(f"Total test images to predict: {len(unique_image_ids)}")

    if sample_limit is not None:
        unique_image_ids = unique_image_ids[:sample_limit]
        sub_df = sub_df[sub_df['image_id'].isin(unique_image_ids)].copy()
        print(f"Limited to {sample_limit} images for fast testing")

    # 3. Setup Test DataLoader
    num_workers = 0 if os.name == 'nt' else 2
    test_dataset = TestDefectDataset(unique_image_ids, test_image_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

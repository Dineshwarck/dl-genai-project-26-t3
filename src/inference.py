import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.model import ScratchUNet
from src.metrics import post_process_mask, CLASS_THRESHOLDS, MIN_AREAS
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

def generate_submission(
    sample_sub_csv='data/sample_submission.csv',
    test_image_dir='data/test',
    checkpoint_path=None,
    wandb_artifact='scratch-unet-best:latest',
    output_csv='submission.csv',
    threshold=CLASS_THRESHOLDS,
    min_areas=MIN_AREAS,
    use_tta=True,
    batch_size=8,
    sample_limit=None,
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on device: {device}")

    # --- 1. Resolve Checkpoint from W&B or Local ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Using local checkpoint: {checkpoint_path}")
    elif wandb_artifact:
        checkpoint_path = fetch_model_from_wandb(artifact_name=wandb_artifact)
    else:
        checkpoint_path = fetch_model_from_wandb(artifact_name='scratch-unet-best:latest')

    # 2. Load Submission Template
    sub_df = pd.read_csv(sample_sub_csv)
    unique_image_ids = sub_df['image_id'].unique()
    print(f"Total test images to predict: {len(unique_image_ids)}")

    # Apply fast testing sample limit if specified
    if sample_limit is not None:
        unique_image_ids = unique_image_ids[:sample_limit]
        sub_df = sub_df[sub_df['image_id'].isin(unique_image_ids)].copy()
        print(f"Limited to {sample_limit} images for fast testing")

    # 3. Setup Test DataLoader
    num_workers = 0 if os.name == 'nt' else 2
    test_dataset = TestDefectDataset(unique_image_ids, test_image_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 4. Load Trained Model
    model = ScratchUNet(in_channels=3, out_channels=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {checkpoint_path}")

    # Setup per-channel threshold tensor
    if isinstance(threshold, (int, float)):
        th_tensor = torch.full((1, 4, 1, 1), float(threshold), device=device)
    elif isinstance(threshold, dict):
        th_tensor = torch.tensor([threshold.get(i, 0.5) for i in range(4)], device=device, dtype=torch.float32).view(1, 4, 1, 1)
    elif isinstance(threshold, (list, tuple)):
        th_tensor = torch.tensor(threshold, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    else:
        th_tensor = torch.full((1, 4, 1, 1), 0.5, device=device)

    # 5. Predict & Post-process
    predictions = {}  # Map (image_id, defect_type) -> rle_string

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc=f"Generating Predictions{' (with TTA)' if use_tta else ''}"):
            images = images.to(device)
            if use_tta:
                p1 = torch.sigmoid(model(images))
                images_hflip = torch.flip(images, dims=[3])
                p2 = torch.flip(torch.sigmoid(model(images_hflip)), dims=[3])
                probs = 0.5 * (p1 + p2)
            else:
                probs = torch.sigmoid(model(images))

            preds_bin = (probs > th_tensor).cpu().numpy().astype(np.uint8)  # (B, 4, 128, 800)

            for i, img_id in enumerate(img_ids):
                # Clean small noise blobs below area thresholds
                cleaned_mask_4ch = post_process_mask(preds_bin[i], min_areas=min_areas)

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

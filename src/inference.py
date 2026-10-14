import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.model import ScratchUNet
from src.metrics import post_process_mask
from src.utils import mask_to_rle
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
    checkpoint_path='models/best_scratch_unet.pth',
    output_csv='submission.csv',
    threshold=0.5,
    batch_size=8,
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on device: {device}")

    # 1. Load Submission Template
    sub_df = pd.read_csv(sample_sub_csv)
    unique_image_ids = sub_df['image_id'].unique()
    print(f"Total test images to predict: {len(unique_image_ids)}")

    # 2. Setup Test DataLoader
    test_dataset = TestDefectDataset(unique_image_ids, test_image_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Load Trained Model
    model = ScratchUNet(in_channels=3, out_channels=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {checkpoint_path}")

    # 4. Predict & Post-process
    predictions = {}  # Map (image_id, defect_type) -> rle_string

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Generating Predictions"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds_bin = (probs > threshold).cpu().numpy().astype(np.uint8)  # (B, 4, 128, 800)

            for i, img_id in enumerate(img_ids):
                # Clean small noise blobs below area thresholds
                cleaned_mask_4ch = post_process_mask(preds_bin[i])

                for c, defect_name in enumerate(DEFECT_CLASSES):
                    channel_mask = cleaned_mask_4ch[c]
                    if channel_mask.sum() > 0:
                        rle_str = mask_to_rle(channel_mask)
                    else:
                        rle_str = " "
                    predictions[(img_id, defect_name)] = rle_str

    # 5. Populate Template
    sub_df['mask_rle'] = sub_df.apply(
        lambda row: predictions.get((row['image_id'], row['defect_type']), " "),
        axis=1
    )

    # 6. Save & Summary
    sub_df.to_csv(output_csv, index=False)
    print(f"\n✅ Submission successfully saved to: {output_csv}")

    # Quick Sanity Check
    non_empty_count = (sub_df['mask_rle'].str.strip() != "").sum()
    print(f"Submission Shape: {sub_df.shape} (Expected: (12000, 4))")
    print(f"Detected Defect Rows: {non_empty_count} / {len(sub_df)} ({non_empty_count/len(sub_df)*100:.2f}%)")
    print(f"Clean Rows: {len(sub_df) - non_empty_count}")


if __name__ == '__main__':
    generate_submission()

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.utils import rle_to_mask

DEFECT_CLASSES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']


class SurfaceDefectDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        """
        Args:
            df: Pandas DataFrame containing either:
                - Raw format: columns ['image_id', 'defect_type', 'mask_rle']
                - Pivoted format: columns ['image_id', 'defect_a', 'defect_b', 'defect_c', 'defect_d']
            image_dir: Path to directory containing the .jpg steel surface images.
            transforms: Albumentations compose transform pipeline.
        """
        self.image_dir = image_dir
        self.transforms = transforms

        # If raw 4-rows-per-image format, pivot to 1 unique row per image
        if 'defect_type' in df.columns:
            self.df = df.pivot(
                index='image_id', columns='defect_type', values='mask_rle'
            ).reset_index()
        else:
            self.df = df.reset_index(drop=True)

    def __len__(self):
        """Returns total unique images."""
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])

        # 1. Load Image
        if not image_id.endswith('.jpg'):
            image_path = os.path.join(self.image_dir, image_id + '.jpg')
        else:
            image_path = os.path.join(self.image_dir, image_id)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # 2. Build 4-Channel Target Mask: shape (H, W, 4)
        mask = np.zeros((h, w, len(DEFECT_CLASSES)), dtype=np.uint8)

        for c, defect in enumerate(DEFECT_CLASSES):
            rle = row.get(defect, np.nan)
            if pd.notna(rle) and str(rle).strip() != '':
                mask[:, :, c] = rle_to_mask(str(rle), height=h, width=w)

        # 3. Apply Synchronized Augmentations (Albumentations)
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 4. Convert NumPy Arrays to PyTorch Tensors
        # Image: (H, W, 3) -> (3, H, W) normalized to [0.0, 1.0]
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Mask: (H, W, 4) -> (4, H, W) binary values {0.0, 1.0}
        mask_tensor = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)

        return image_tensor, mask_tensor


def get_stratified_split(df, val_size=0.20, random_state=42):
    """
    Creates an 80/20 train/validation split stratified by multi-label defect combination signature
    (from Milestone 1 Question 11 findings), ensuring rare defects like defect_b are balanced.
    Implemented in pure pandas without external dependencies.
    
    Returns:
        train_df, val_df (pivoted format: 1 row per unique image)
    """
    if 'defect_type' in df.columns:
        pivoted = df.pivot(
            index='image_id', columns='defect_type', values='mask_rle'
        ).reset_index()
    else:
        pivoted = df.copy()

    # Generate combination signature per image
    def make_signature(row):
        active = [c for c in DEFECT_CLASSES if pd.notna(row.get(c)) and str(row.get(c)).strip() != '']
        if len(active) == 0:
            return 'clean'
        return '+'.join(sorted(active))

    pivoted['strat_signature'] = pivoted.apply(make_signature, axis=1)

    # Stratify by sampling each combination signature group in exact proportion
    train_df = (
        pivoted.groupby('strat_signature', group_keys=False)
        .apply(lambda group: group.sample(frac=1.0 - val_size, random_state=random_state))
    )
    val_df = pivoted.drop(train_df.index)

    train_df = train_df.drop(columns=['strat_signature'], errors='ignore').reset_index(drop=True)
    val_df = val_df.drop(columns=['strat_signature'], errors='ignore').reset_index(drop=True)

    return train_df, val_df

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils import rle_to_mask

class SurfaceDefectDataset(Dataset):
    def __init__(self,df,image_dir,transforms=None):
        """
        df: Pandas dataframe containing 'image_id' and 'mask_rle'
        image_dir: Path to the folder containing the raw images
        transforms: Albumentations transforms to apply

        """
        self.df=df
        self.image_dir=image_dir
        self.transforms=transforms
    
    def __len__(self):
        # Return the total number of rows in the dataframe"
        return len(self.df)
    
    def __getitem__(self,idx):
        # 1. Get the row at 'idx'
        row = self.df.iloc[idx]
        image_id= row['image_id']
        rle = row['mask_rle']

        # 2. Construct the full image path and load it with cv2
        if not image_id.endswith('.jpg'):
            image_path = os.path.join(self.image_dir,image_id + '.jpg')
        image = cv2.imread(image_path)

        # cv2 reads images in BGR format, convert to RGB
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        # 3. Get the mask using rle_to_mask function
        mask = rle_to_mask(rle,height=128,width=800)
        
        # 4. Apply augmentation if provided
        if self.transforms:
            augmented = self.transforms(image=image,mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 5. Convert NumPy arrays to PyTorch Tensors
        # Images need to be reshaped from (H,W,C) to (C,H,W) for PyTorch
        image_tensor = torch.tensor(image,dtype=torch.float32).permute(2,0,1)  

        # Masks should have an extra channel dimension: (1,H,W)
        mask_tensor = torch.tensor(mask,dtype=torch.float32).unsqueeze(0)

        # Normalize image to [0,1] if not handled by transforms
        image_tensor = image_tensor/255.0
        
        return image_tensor,mask_tensor


        
        




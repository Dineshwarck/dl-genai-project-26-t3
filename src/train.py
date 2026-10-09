import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
import pandas as pd
import wandb

# Import Scratch model and dataset
from src.dataset import SurfaceDefectDataset
from src.model import ScratchUNet

def get_augmentations():
    # 1. Defining Albumentations (We need to resize for the model to fit in memory)
    # Let's resize to 128x800 for the baseline to keep it fast

    train_transform = A.Compose([
        A.Resize(height=128, width=800),
        A.HorizontalFlip(p=0.5)
    ])
   
    return train_transform

def train_baseline():
    # 2. Initializing Weights & biases for training
    wandb.init(project="24F2004781-t32026", name="scratch-unet-baseline")

    # 3. Device Agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 4. Load Data and create DataLoaders
    train_df = pd.read_csv(r"D:\SurfaceGuard_AI\data\train.csv")  

    # Using the first 1000 rows to test the pipeline quickly
    train_df = train_df.dropna().head(1000) 
    
    dataset = SurfaceDefectDataset(
        df=train_df,
        image_dir=r"D:\SurfaceGuard_AI\data\train",
        transforms=get_augmentations()  
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    # 5. Initialize Model, Loss, and Optimizer
    model = ScratchUNet(in_channels=3,out_channels=1).to(device)

    # BCEWithLogitsLoss is best for multi-label binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.01)

    # 6. The Training Loop (Just 2 epochs for the baseline test)
    epochs = 2

    for epoch in range(epochs):
        model.train()
        epoch_loss=0

        for images,masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward Pass
            predictions = model(images)

            # Calculate loss
            loss = criterion(predictions,masks)

            # Backward pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(dataloader)       
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        wandb.log({"train_loss":avg_loss,"epoch":epoch+1})

    print("Baseline Training Complete!")
    wandb.finish()

if __name__=="__main__":
    train_baseline()
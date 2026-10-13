import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits) # (BxCxHxW)
        # Flatten spatial dimensions of probs and targets:
        # e.g probs.view(B,C,-1) and targets.view(B,C,-1)
        probs_flat = probs.flatten(2)
        targets_flat = targets.flatten(2)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()

class ComboLoss(nn.Module):
    def __init__(self,bce_weight=1.0, dice_weight = 1.0, smooth = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()        
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)       
        dice_loss = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss
    
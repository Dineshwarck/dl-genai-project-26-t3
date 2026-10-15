import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, channel_weights=None):
        """
        Soft Dice Loss supporting uniform or per-channel weighting.
        """
        super().__init__()
        self.smooth = smooth
        if channel_weights is not None:
            if not isinstance(channel_weights, torch.Tensor):
                channel_weights = torch.tensor(channel_weights, dtype=torch.float32)
        self.register_buffer('channel_weights', channel_weights)
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits) # (B, C, H, W)
        # Flatten spatial dimensions:
        probs_flat = probs.flatten(2)
        targets_flat = targets.flatten(2)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        if self.channel_weights is not None:
            cw = self.channel_weights.to(device=dice.device, dtype=dice.dtype)
            weighted_dice = (dice * cw).sum(dim=1) / (cw.sum() + 1e-7)
            return 1.0 - weighted_dice.mean()

        return 1.0 - dice.mean()

class ComboLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        smooth=1.0,
        pos_weight=(12.0, 25.0, 3.0, 6.0),
        channel_weights=(1.5, 2.5, 1.0, 1.2),
    ):
        """
        ComboLoss combining BCEWithLogitsLoss with positive pixel weighting (pos_weight)
        and SoftDiceLoss with optional per-channel weighting.

        Args:
            bce_weight: Weight multiplier for BCE loss.
            dice_weight: Weight multiplier for Dice loss.
            smooth: Smoothing factor for Dice calculation.
            pos_weight: Per-channel weight applied to positive (defect) pixels in BCE
                        to combat extreme background class imbalance:
                        - defect_a (small pits, 1440:1 neg/pos): 12.0
                        - defect_b (scratches, 5408:1 neg/pos): 25.0
                        - defect_c (crazing, 50:1 neg/pos): 3.0
                        - defect_d (patches, 95:1 neg/pos): 6.0
            channel_weights: Relative importance weights for each defect channel in Dice loss:
                             [1.5, 2.5, 1.0, 1.2] to prioritize rare classes over dominant defect_c.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                if isinstance(pos_weight, (int, float)):
                    pos_weight = torch.tensor([float(pos_weight)])
                else:
                    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            if pos_weight.ndim == 1 and len(pos_weight) > 1:
                pos_weight = pos_weight.view(1, -1, 1, 1)
        self.register_buffer('pos_weight', pos_weight)

        self.dice = SoftDiceLoss(smooth=smooth, channel_weights=channel_weights)

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            pw = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        dice_loss = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss

    
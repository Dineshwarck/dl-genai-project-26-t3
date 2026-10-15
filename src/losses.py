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
        probs = torch.sigmoid(logits)  # (B, C, H, W)
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Multi-label Binary Focal Loss.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Dramatically down-weights easy background pixels ((1 - p_t)^2 ≈ 0.00001),
        preventing clean steel from drowning out rare defects (defect_a and defect_b).

        Args:
            alpha: Balancing factor for positive vs negative pixels per channel.
                   Default: (0.75, 0.85, 0.50, 0.60)
            gamma: Focusing parameter (default 2.0).
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                if isinstance(alpha, (int, float)):
                    alpha = torch.tensor([float(alpha)])
                else:
                    alpha = torch.tensor(alpha, dtype=torch.float32)
            if alpha.ndim == 1 and len(alpha) > 1:
                alpha = alpha.view(1, -1, 1, 1)
        self.register_buffer('alpha', alpha)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t: probability of the ground-truth class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            a = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = targets * a + (1.0 - targets) * (1.0 - a)
            focal_loss = alpha_t * focal_weight * bce
        else:
            focal_loss = focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ComboLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        smooth=1.0,
        use_focal=True,
        gamma=2.0,
        alpha=(0.75, 0.85, 0.50, 0.60),
        pos_weight=(12.0, 25.0, 3.0, 6.0),
        channel_weights=(1.5, 2.5, 1.0, 1.2),
    ):
        """
        ComboLoss combining Focal Loss (or BCEWithLogitsLoss) and SoftDiceLoss.

        Args:
            bce_weight: Weight multiplier for the pixel loss (Focal or BCE).
            dice_weight: Weight multiplier for the region loss (Dice).
            smooth: Smoothing factor for Dice calculation.
            use_focal: If True, uses FocalLoss with gamma and alpha (recommended).
                       If False, uses BCEWithLogitsLoss with pos_weight.
            gamma: Focusing parameter for Focal Loss (default 2.0).
            alpha: Class balancing factor for Focal Loss (default [0.75, 0.85, 0.50, 0.60]).
            pos_weight: Per-channel weight for standard BCE fallback.
            channel_weights: Relative importance weights for each defect channel in Dice loss.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_focal = use_focal

        if use_focal:
            self.pixel_loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            if pos_weight is not None:
                if not isinstance(pos_weight, torch.Tensor):
                    if isinstance(pos_weight, (int, float)):
                        pos_weight = torch.tensor([float(pos_weight)])
                    else:
                        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
                if pos_weight.ndim == 1 and len(pos_weight) > 1:
                    pos_weight = pos_weight.view(1, -1, 1, 1)
            self.register_buffer('pos_weight', pos_weight)
            self.pixel_loss_fn = None

        self.dice = SoftDiceLoss(smooth=smooth, channel_weights=channel_weights)

    def forward(self, logits, targets):
        if self.use_focal:
            # Scale focal loss by 5.0 to match dice loss magnitude (~0.5 - 1.0)
            pixel_loss = 5.0 * self.pixel_loss_fn(logits, targets)
        else:
            if self.pos_weight is not None:
                pw = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
                pixel_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
            else:
                pixel_loss = F.binary_cross_entropy_with_logits(logits, targets)

        dice_loss = self.dice(logits, targets)
        return self.bce_weight * pixel_loss + self.dice_weight * dice_loss

    
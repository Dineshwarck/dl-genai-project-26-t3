import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, channel_weights=None, mask_empty=True):
        """
        Soft Dice Loss supporting uniform or per-channel weighting with empty-slice masking.

        Args:
            smooth: Smoothing factor to avoid division by zero.
            channel_weights: Relative weights per defect channel.
            mask_empty: If True (default), computes Dice loss only on (batch, channel) slices
                        that contain ground-truth defect annotations (targets.sum() > 0).
                        This prevents the 'Empty-Slice Dice Gradient Pitfall', where rare classes
                        (defect_a and defect_b) are forced to negative infinity logits because
                        98% of images are empty. Focal/BCE loss handles false positive suppression.
        """
        super().__init__()
        self.smooth = smooth
        self.mask_empty = mask_empty
        if channel_weights is not None:
            if not isinstance(channel_weights, torch.Tensor):
                channel_weights = torch.tensor(channel_weights, dtype=torch.float32)
        self.register_buffer('channel_weights', channel_weights)
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)  # (B, C, H, W)
        probs_flat = probs.flatten(2)
        targets_flat = targets.flatten(2)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        target_sum = targets_flat.sum(dim=2)
        union = probs_flat.sum(dim=2) + target_sum

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.mask_empty:
            has_gt = target_sum > 0  # (B, C)
            if has_gt.any():
                if self.channel_weights is not None:
                    cw = self.channel_weights.to(device=dice.device, dtype=dice.dtype)
                    cw_expanded = cw.unsqueeze(0).expand_as(dice)
                    valid_weights = cw_expanded * has_gt.float()
                    dice_loss = 1.0 - (dice * valid_weights).sum() / (valid_weights.sum() + 1e-7)
                else:
                    dice_loss = 1.0 - dice[has_gt].mean()
                return dice_loss
            else:
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
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


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension with respect to sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-7)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss on flattened 1D logits and labels."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszLoss(nn.Module):
    """Multi-label Lovasz-Hinge loss directly optimizing the Jaccard/Dice surrogate
    at the continuous level, penalizing fuzzy boundary transitions and edge bleeding.
    """
    def __init__(self, per_image=False):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        total_loss = 0.0
        if self.per_image:
            count = 0
            for c in range(C):
                for b in range(B):
                    total_loss = total_loss + lovasz_hinge_flat(logits[b, c].reshape(-1), targets[b, c].reshape(-1))
                    count += 1
            return total_loss / max(count, 1)
        else:
            for c in range(C):
                l_flat = logits[:, c].reshape(-1)
                t_flat = targets[:, c].reshape(-1)
                total_loss = total_loss + lovasz_hinge_flat(l_flat, t_flat)
            return total_loss / C


class ComboLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        lovasz_weight=0.5,
        use_lovasz=True,
        smooth=1.0,
        use_focal=True,
        gamma=2.0,
        alpha=(0.75, 0.85, 0.50, 0.60),
        pos_weight=(12.0, 25.0, 3.0, 6.0),
        channel_weights=(1.5, 2.5, 1.0, 1.2),
        mask_empty=True,
    ):
        """
        ComboLoss combining Focal Loss, SoftDiceLoss, and continuous LovaszLoss.

        Args:
            bce_weight: Weight multiplier for the pixel loss (Focal or BCE).
            dice_weight: Weight multiplier for the region loss (Dice).
            lovasz_weight: Weight multiplier for the boundary Lovasz-hinge loss.
            use_lovasz: If True, incorporates Lovasz-hinge loss for sharp defect edges.
            smooth: Smoothing factor for Dice calculation.
            use_focal: If True, uses FocalLoss with gamma and alpha (recommended).
                       If False, uses BCEWithLogitsLoss with pos_weight.
            gamma: Focusing parameter for Focal Loss (default 2.0).
            alpha: Class balancing factor for Focal Loss (default [0.75, 0.85, 0.50, 0.60]).
            pos_weight: Per-channel weight for standard BCE fallback.
            channel_weights: Relative importance weights for each defect channel in Dice loss.
            mask_empty: If True (default), computes Dice loss only on slices with positive targets.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.use_lovasz = use_lovasz
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

        self.dice = SoftDiceLoss(smooth=smooth, channel_weights=channel_weights, mask_empty=mask_empty)
        self.lovasz = LovaszLoss(per_image=False)

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
        total_loss = self.bce_weight * pixel_loss + self.dice_weight * dice_loss
        if self.use_lovasz:
            total_loss = total_loss + self.lovasz_weight * self.lovasz(logits, targets)
        return total_loss

    
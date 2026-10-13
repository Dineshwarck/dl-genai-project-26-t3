import cv2
import numpy as np
import torch

# Minimum area thresholds derived from Milestone 1 Q10
MIN_AREAS = {
    0:15, # defect_a: tiny pits (keep threshold small!)
    1:50, # defect_b: scratches (safe threshold N=50)
    2:50, # defect_c: crazing
    3:50, # defect_d: patches
}

def post_process_mask(mask_numpy, min_areas=MIN_AREAS):
    """
    mask_numpy: shape (4, H, W) binary {0,1}
    Filters small connected blobs below class thresholds.
    """
    clean_mask = np.zeros_like(mask_numpy)
    num_classes = mask_numpy.shape[0]
    for cls_idx in range(num_classes):
        channel = mask_numpy[cls_idx].astype(np.uint8)
        min_area = min_areas.get(cls_idx,0)
        if channel.sum() == 0 or min_area == 0:
            clean_mask[cls_idx] = channel
            continue
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            channel, connectivity=8
        ) 
    
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                clean_mask[cls_idx][labels == label] = 1
    return clean_mask

def compute_competition_metrics(pred_masks, gt_masks, threshold=0.5):
  """pred_masks: torch.Tensor, shape (B, 4, H, W) - logits or binary masks.

  gt_masks:   torch.Tensor, shape (B, 4, H, W) - ground truth binary {0, 1}.
  threshold:  float, decision threshold if pred_masks are logits.
  """
  N, C, H, W = pred_masks.shape
  canvas_pixels = H * W

  # If logits, binarize with sigmoid + threshold
  if pred_masks.dtype in (torch.float32, torch.float64):
    if pred_masks.min() < 0.0 or pred_masks.max() > 1.0:
      preds_bin = (torch.sigmoid(pred_masks) > threshold).float()
    else:
      preds_bin = (pred_masks > threshold).float()
  else:
    preds_bin = pred_masks.float()

  pred_flat = preds_bin.reshape(N * C, -1)
  gt_flat = gt_masks.float().reshape(N * C, -1)

  pos_mask = gt_flat.sum(dim=1) > 0
  neg_mask = ~pos_mask

  # 1. Foreground Dice strictly across positive pairs
  foreground_dice = 0.0
  if pos_mask.any():
    intersection = (pred_flat[pos_mask] * gt_flat[pos_mask]).sum(dim=1)
    union = pred_flat[pos_mask].sum(dim=1) + gt_flat[pos_mask].sum(dim=1)
    dice = (2.0 * intersection) / (union + 1e-7)
    foreground_dice = dice.mean().item()

  # 2. False Positive Score & Empty Accuracy across empty pairs
  false_positive_score = 1.0
  empty_accuracy = 1.0
  if neg_mask.any():
    fp_pixels = pred_flat[neg_mask].sum(dim=1)
    fp_frac = fp_pixels / canvas_pixels
    false_positive_score = (1.0 - fp_frac.mean()).item()
    empty_accuracy = (fp_pixels == 0).float().mean().item()

  # 3. Official Composite Score (Milestone 1, Question 8)
  official_score = (
      (0.70 * foreground_dice)
      + (0.20 * false_positive_score)
      + (0.10 * empty_accuracy)
  )

  return {
      'foreground_dice': foreground_dice,
      'false_positive_score': false_positive_score,
      'empty_accuracy': empty_accuracy,
      'official_score': official_score,
  }

    

    
    

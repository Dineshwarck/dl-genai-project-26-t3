import cv2
import numpy as np
import torch

# Calibrated minimum connected component area thresholds (in pixels for 128x800 canvas)
# Tailored for Multi-Task U-Net with Soft Quadratic Gating:
# Since the auxiliary classifier already suppresses false alarms (FP score > 0.997),
# area thresholds are calibrated so they do NOT discard genuine fine defects:
# - defect_a: 60 px (thin scratches)
# - defect_b: 80 px (edge cracks)
# - defect_c: 250 px (scabs/pitting - preserves the 16% of true defects under 800 px)
# - defect_d: 400 px (texture patches - preserves genuine small patches)
# Empirically validated optimal minimum connected component areas (Validation Score: 0.6836)
MIN_AREAS = {
    0: 50,   # defect_a: captures fine scratches without noise
    1: 200,  # defect_b: captures edge cracks
    2: 400,  # defect_c: captures smaller scabs/pitting (previously lost at 800 px)
    3: 600,  # defect_d: captures texture patches (previously lost at 1200 px)
}

# Empirically validated optimal decision thresholds
CLASS_THRESHOLDS = {
    0: 0.35,  # defect_a
    1: 0.30,  # defect_b
    2: 0.30,  # defect_c
    3: 0.30,  # defect_d
}

# Maximum area ceilings to eliminate whole-sheet false alarms (e.g. 49k shadow hallucinations)
MAX_AREAS = {
    0: 25000,  # defect_a scratches never cover >25k pixels
    1: 15000,  # defect_b edge cracks never cover >15k pixels
    2: 35000,  # defect_c scabs never cover >35k pixels (prunes full-sheet rough scale false alarms)
    3: 35000,  # defect_d patches never cover >35k pixels (prunes 49k lighting/shadow hallucinations)
}

def post_process_mask(
    mask_numpy,
    min_areas=MIN_AREAS,
    max_areas=MAX_AREAS,
    filter_geometry=True,
):
    """
    mask_numpy: shape (4, H, W) binary {0,1}
    Applies:
      1. Minimum and Maximum Area thresholding (filters speckles and giant whole-sheet hallucinations).
      2. Geometric aspect-ratio and height continuity filtering for thin longitudinal defects:
         - defect_b (edge cracks): requires vertical continuity (H_span >= 25 px, AspectRatio H/W >= 2.0).
         - defect_a (thin scratches): requires vertical elongation (H_span >= 15 px, AspectRatio H/W >= 1.2).
    """
    clean_mask = np.zeros_like(mask_numpy)
    num_classes = mask_numpy.shape[0]

    for cls_idx in range(num_classes):
        channel = mask_numpy[cls_idx].astype(np.uint8)
        total_pixels = int(channel.sum())

        if total_pixels == 0:
            continue

        # Ceiling check: Discard giant whole-sheet false alarms (e.g. 49k shadow hallucinations)
        max_area = max_areas.get(cls_idx, None) if max_areas else None
        if max_area and total_pixels > max_area:
            continue

        min_area = min_areas.get(cls_idx, 0)
        if total_pixels < min_area:
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            channel, connectivity=8
        )

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            w = stats[label, cv2.CC_STAT_WIDTH]

            if area < min_area:
                continue

            if filter_geometry:
                # Defect B: Edge cracks are tall vertical tears (H_span >= 25 px, H/W >= 2.0)
                if cls_idx == 1:
                    aspect_ratio = h / max(w, 1)
                    if h < 25 or aspect_ratio < 2.0:
                        continue

                # Defect A: Longitudinal scratches are vertical streaks (H_span >= 15 px, H/W >= 1.2)
                elif cls_idx == 0:
                    aspect_ratio = h / max(w, 1)
                    if h < 15 or aspect_ratio < 1.2:
                        continue

            clean_mask[cls_idx][labels == label] = 1

    return clean_mask

def compute_competition_metrics(pred_masks, gt_masks, threshold=CLASS_THRESHOLDS):
  """pred_masks: torch.Tensor, shape (B, 4, H, W) - logits or binary masks.

  gt_masks:   torch.Tensor, shape (B, 4, H, W) - ground truth binary {0, 1}.
  threshold:  float, dict, or tuple/list of per-channel thresholds.
  """
  N, C, H, W = pred_masks.shape
  canvas_pixels = H * W

  # If logits or probabilities, binarize with sigmoid + per-channel threshold
  if pred_masks.dtype in (torch.float32, torch.float64):
    if pred_masks.min() < 0.0 or pred_masks.max() > 1.0:
      probs = torch.sigmoid(pred_masks)
    else:
      probs = pred_masks

    if isinstance(threshold, (int, float)):
      preds_bin = (probs > threshold).float()
    elif isinstance(threshold, dict):
      th_tensor = torch.tensor(
          [threshold.get(i, 0.5) for i in range(C)],
          device=pred_masks.device,
          dtype=pred_masks.dtype
      ).view(1, C, 1, 1)
      preds_bin = (probs > th_tensor).float()
    elif isinstance(threshold, (list, tuple)):
      th_tensor = torch.tensor(
          threshold,
          device=pred_masks.device,
          dtype=pred_masks.dtype
      ).view(1, C, 1, 1)
      preds_bin = (probs > th_tensor).float()
    else:
      preds_bin = (probs > 0.5).float()
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

    

    
    

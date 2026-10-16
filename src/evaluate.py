import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import SurfaceDefectDataset, get_stratified_split, DEFECT_CLASSES
from src.model import ScratchUNet
from src.metrics import post_process_mask, compute_competition_metrics, CLASS_THRESHOLDS, MIN_AREAS
from src.utils import mask_to_rle, rle_to_mask
from src.inference import fetch_model_from_wandb


def evaluate_validation_set(
    data_csv='data/train.csv',
    image_dir='data/train',
    checkpoint_path='models/best_scratch_unet.pth',
    wandb_artifact='scratch-unet-best:latest',
    output_csv='val_predictions.csv',
    val_size=0.20,
    random_state=42,
    batch_size=8,
    threshold=CLASS_THRESHOLDS,
    min_areas=MIN_AREAS,
    sample_limit=None,
    device=None,
):
    """
    Evaluates the model on the stratified validation holdout set and generates
    a comprehensive per-sample CSV comparison of Ground Truth vs Predictions.

    Args:
        data_csv: Path to train.csv containing annotations.
        image_dir: Path to directory containing train images.
        checkpoint_path: Path to model checkpoint file (.pth).
        wandb_artifact: W&B model artifact name to fetch if checkpoint_path doesn't exist.
        output_csv: Path to save the detailed validation predictions CSV.
        val_size: Validation holdout proportion (default 0.20).
        random_state: Stratification random seed (default 42).
        batch_size: Batch size for inference.
        threshold: Decision threshold (float or dict/list per defect class).
        sample_limit: Optional limit on validation images for fast dry-runs.
        device: 'cuda' or 'cpu' (auto-detected if None).

    Returns:
        results_df: Pandas DataFrame containing per-defect GT and prediction comparisons.
        metrics_dict: Dictionary of competition metrics on the validation set.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running validation evaluation on device: {device}")

    # 1. Resolve Checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Using local checkpoint: {checkpoint_path}")
    elif wandb_artifact:
        checkpoint_path = fetch_model_from_wandb(artifact_name=wandb_artifact)
    else:
        checkpoint_path = fetch_model_from_wandb(artifact_name='scratch-unet-best:latest')

    # 2. Load Stratified Validation Split
    df = pd.read_csv(data_csv)
    _, val_df = get_stratified_split(df, val_size=val_size, random_state=random_state)

    if sample_limit is not None:
        val_df = val_df.head(sample_limit)
        print(f"Limited to {sample_limit} validation images for fast evaluation")
    print(f"Total validation images to evaluate: {len(val_df)}")

    val_dataset = SurfaceDefectDataset(val_df, image_dir=image_dir, transforms=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
    )

    # 3. Load Trained Model
    model = ScratchUNet(in_channels=3, out_channels=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {checkpoint_path}")

    # 4. Setup Threshold Tensor
    if isinstance(threshold, (int, float)):
        th_tensor = torch.full((1, 4, 1, 1), float(threshold), device=device)
    elif isinstance(threshold, dict):
        th_tensor = torch.tensor([threshold.get(i, 0.5) for i in range(4)], device=device, dtype=torch.float32).view(1, 4, 1, 1)
    elif isinstance(threshold, (list, tuple)):
        th_tensor = torch.tensor(threshold, device=device, dtype=torch.float32).view(1, 4, 1, 1)
    else:
        th_tensor = torch.full((1, 4, 1, 1), 0.5, device=device)

    # 5. Evaluate Batches
    records = []
    all_preds_list = []
    all_targets_list = []

    img_counter = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating Validation Set"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            preds_bin = (probs > th_tensor).cpu().numpy().astype(np.uint8)
            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy().astype(np.uint8)

            all_preds_list.append(logits.cpu())
            all_targets_list.append(masks.cpu())

            batch_len = images.shape[0]
            for b in range(batch_len):
                row = val_df.iloc[img_counter]
                image_id = str(row['image_id'])
                img_counter += 1

                cleaned_mask_4ch = post_process_mask(preds_bin[b], min_areas=min_areas)

                for c, defect_name in enumerate(DEFECT_CLASSES):
                    gt_ch = masks_np[b, c]
                    pred_ch = cleaned_mask_4ch[c]
                    prob_ch = probs_np[b, c]

                    gt_pixels = int(gt_ch.sum())
                    pred_pixels = int(pred_ch.sum())
                    intersection = int((gt_ch * pred_ch).sum())
                    max_prob = float(prob_ch.max())

                    # Calculate Dice
                    if gt_pixels > 0 and pred_pixels > 0:
                        dice = (2.0 * intersection) / (gt_pixels + pred_pixels)
                        error_type = 'TP' if dice >= 0.50 else 'Partial_TP'
                    elif gt_pixels == 0 and pred_pixels == 0:
                        dice = 1.0
                        error_type = 'TN'
                    elif gt_pixels > 0 and pred_pixels == 0:
                        dice = 0.0
                        error_type = 'FN'  # Missed defect
                    else:  # gt_pixels == 0 and pred_pixels > 0
                        dice = 0.0
                        error_type = 'FP'  # Hallucination

                    gt_rle = mask_to_rle(gt_ch) if gt_pixels > 0 else " "
                    pred_rle = mask_to_rle(pred_ch) if pred_pixels > 0 else " "

                    records.append({
                        'image_id': image_id,
                        'defect_type': defect_name,
                        'error_type': error_type,
                        'dice': round(dice, 4),
                        'max_prob': round(max_prob, 4),
                        'gt_pixels': gt_pixels,
                        'pred_pixels': pred_pixels,
                        'gt_rle': gt_rle,
                        'pred_rle': pred_rle,
                    })

    # 6. Build and Save Results DataFrame
    results_df = pd.DataFrame(records)
    results_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Validation predictions saved to: {output_csv}")

    # 7. Compute Official Competition Metrics
    all_preds_tensor = torch.cat(all_preds_list, dim=0)
    all_targets_tensor = torch.cat(all_targets_list, dim=0)
    metrics = compute_competition_metrics(all_preds_tensor, all_targets_tensor, threshold=0.5)

    # Per-Class Foreground Dice Breakdown
    per_class_f1 = {}
    for c, defect_name in enumerate(DEFECT_CLASSES):
        class_df = results_df[(results_df['defect_type'] == defect_name) & (results_df['gt_pixels'] > 0)]
        if len(class_df) > 0:
            per_class_f1[defect_name] = round(class_df['dice'].mean(), 4)
        else:
            per_class_f1[defect_name] = 0.0

    print("\n" + "=" * 50)
    print("       VALIDATION EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Official Competition Score: {metrics['official_score']:.4f}")
    print(f"Foreground Dice (F1):       {metrics['foreground_dice']:.4f}")
    print(f"False Positive Score:       {metrics['false_positive_score']:.4f}")
    print(f"Empty Pair Accuracy:        {metrics['empty_accuracy']:.4f}")
    print("-" * 50)
    print("Per-Class Foreground Dice Breakdown:")
    for defect_name, f1 in per_class_f1.items():
        count = len(results_df[(results_df['defect_type'] == defect_name) & (results_df['gt_pixels'] > 0)])
        print(f"  {defect_name:<10}: F1 = {f1:.4f} (on {count} positive images)")
    print("-" * 50)
    print("Error Type Counts:")
    print(results_df['error_type'].value_counts().to_string())
    print("=" * 50)

    metrics['per_class_f1'] = per_class_f1
    return results_df, metrics


def visualize_defect_comparison(
    image_dir,
    image_id,
    defect_type,
    gt_rle,
    pred_rle,
    dice=None,
    max_prob=None,
    error_type=None,
    save_path=None,
):
    """
    Draws a 4-panel comparison for visual error analysis:
    Panel 1: Raw Steel Image
    Panel 2: Ground Truth Mask Overlay
    Panel 3: Model Prediction Mask Overlay
    Panel 4: Discrepancy Map (Green = GT, Red = Prediction, Yellow = Overlap)
    """
    # Load raw image
    if not image_id.endswith('.jpg'):
        img_path = os.path.join(image_dir, image_id + '.jpg')
    else:
        img_path = os.path.join(image_dir, image_id)

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (800, 128))
    h, w, _ = image.shape

    # Decode RLE masks
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if pd.notna(gt_rle) and str(gt_rle).strip() != '':
        gt_mask = rle_to_mask(str(gt_rle), height=h, width=w)

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    if pd.notna(pred_rle) and str(pred_rle).strip() != '':
        pred_mask = rle_to_mask(str(pred_rle), height=h, width=w)

    fig, axes = plt.subplots(4, 1, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.4)

    title_suffix = ""
    if error_type:
        title_suffix += f" | Type: {error_type}"
    if dice is not None:
        title_suffix += f" | Dice: {dice:.4f}"
    if max_prob is not None:
        title_suffix += f" | Max Prob: {max_prob:.4f}"

    # Panel 1: Original Image
    axes[0].imshow(image)
    axes[0].set_title(f"Image: {image_id} | Class: {defect_type}{title_suffix}", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Ground Truth Overlay
    gt_overlay = image.copy()
    if gt_mask.sum() > 0:
        gt_overlay[gt_mask == 1] = [0, 255, 0]  # Green for Ground Truth
    axes[1].imshow(gt_overlay)
    axes[1].set_title(f"Ground Truth Mask (Pixels: {int(gt_mask.sum())})", fontsize=11, color='green')
    axes[1].axis('off')

    # Panel 3: Model Prediction Overlay
    pred_overlay = image.copy()
    if pred_mask.sum() > 0:
        pred_overlay[pred_mask == 1] = [255, 50, 50]  # Red for Prediction
    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"Model Predicted Mask (Pixels: {int(pred_mask.sum())})", fontsize=11, color='red')
    axes[2].axis('off')

    # Panel 4: Discrepancy Map
    diff_map = image.copy() // 2  # Dim background slightly
    # Green = False Negative (GT only, missed by model)
    fn_mask = (gt_mask == 1) & (pred_mask == 0)
    diff_map[fn_mask] = [0, 255, 0]

    # Red = False Positive (Model only, hallucination)
    fp_mask = (gt_mask == 0) & (pred_mask == 1)
    diff_map[fp_mask] = [255, 0, 0]

    # Yellow = True Positive (Overlap)
    tp_mask = (gt_mask == 1) & (pred_mask == 1)
    diff_map[tp_mask] = [255, 255, 0]

    axes[3].imshow(diff_map)
    axes[3].set_title("Discrepancy: Green = Missed (FN) | Red = Hallucination (FP) | Yellow = Correct (TP)", fontsize=11)
    axes[3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_class_error_gallery(
    val_predictions_csv_or_df,
    image_dir,
    classes=DEFECT_CLASSES,
    samples_per_category=1,
    save_dir=None,
):
    """
    Plots a diagnostic visual gallery of wrong predictions for every defect class:
    1. Worst False Negatives (Missed defects with largest ground truth area)
    2. Worst False Positives (Hallucinations on clean steel with largest predicted area)

    Args:
        val_predictions_csv_or_df: Path to val_predictions.csv or loaded DataFrame.
        image_dir: Path to directory containing raw images.
        classes: List of defect class names to inspect.
        samples_per_category: Number of worst samples to plot per error type.
        save_dir: Optional directory to save output PNG plots instead of displaying inline.
    """
    if isinstance(val_predictions_csv_or_df, str):
        df = pd.read_csv(val_predictions_csv_or_df)
    else:
        df = val_predictions_csv_or_df.copy()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for defect_name in classes:
        print(f"\n" + "=" * 70)
        print(f"       VISUAL ERROR ANALYSIS: {defect_name.upper()}")
        print("=" * 70)

        # 1. Missed Defects (False Negatives: GT > 0, Dice == 0)
        fns = df[(df['defect_type'] == defect_name) & (df['error_type'] == 'FN')].sort_values(
            by='gt_pixels', ascending=False
        ).head(samples_per_category)

        if len(fns) > 0:
            print(f"\n>>> Worst False Negative (Missed {defect_name}):")
            for _, row in fns.iterrows():
                save_path = os.path.join(save_dir, f"{defect_name}_FN_{row['image_id']}.png") if save_dir else None
                visualize_defect_comparison(
                    image_dir=image_dir,
                    image_id=row['image_id'],
                    defect_type=defect_name,
                    gt_rle=row['gt_rle'],
                    pred_rle=row['pred_rle'],
                    dice=row['dice'],
                    max_prob=row['max_prob'],
                    error_type="False Negative (Missed Defect)",
                    save_path=save_path,
                )
        else:
            print(f"No False Negatives found for {defect_name}.")

        # 2. Hallucinations (False Positives: GT == 0, Pred > 0)
        fps = df[(df['defect_type'] == defect_name) & (df['error_type'] == 'FP')].sort_values(
            by='pred_pixels', ascending=False
        ).head(samples_per_category)

        if len(fps) > 0:
            print(f"\n>>> Worst False Positive (Hallucination on Clean Steel for {defect_name}):")
            for _, row in fps.iterrows():
                save_path = os.path.join(save_dir, f"{defect_name}_FP_{row['image_id']}.png") if save_dir else None
                visualize_defect_comparison(
                    image_dir=image_dir,
                    image_id=row['image_id'],
                    defect_type=defect_name,
                    gt_rle=row['gt_rle'],
                    pred_rle=row['pred_rle'],
                    dice=row['dice'],
                    max_prob=row['max_prob'],
                    error_type="False Positive (Hallucination)",
                    save_path=save_path,
                )
        else:
            print(f"No False Positives found for {defect_name}.")

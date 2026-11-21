# SurfaceGuard AI: Industrial Steel Surface Defect Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-24F2004781--t32026-FFBE00.svg?style=flat&logo=weightsandbiases)](https://wandb.ai/dineshwar-developer-indian-institute-of-technology/24F2004781-t32026)
[![Kaggle Score](https://img.shields.io/badge/Kaggle_Public_Score-0.65905-20BEFF.svg?style=flat&logo=kaggle)](https://www.kaggle.com)

**SurfaceGuard AI** is an industrial-grade computer vision pipeline developed for automated, pixel-level surface defect detection on manufactured steel sheets. Operating at a native rectangular canvas of $128 \times 800$ (Aspect Ratio $1:6.25$), the system identifies and segments four distinct defect categories (**Defect A: Pitting**, **Defect B: Longitudinal Tears**, **Defect C: Parallel Crazing Scratches**, and **Defect D: Patch Inclusions**) under severe class imbalance and extreme penalties for false alarms.

---

## 1. Project Directory Structure

```text
SurfaceGuard_AI/
│
├── .env                                # Local secrets & environment configuration (WANDB_API_KEY) [git-ignored]
├── .gitignore                          # Git ignore specification for data, checkpoints, and credentials
├── README.md                           # Master project documentation, architecture overview, and reproduction guide
├── requirements.txt                    # Python runtime package dependencies
│
├── data/                               # Dataset directory (Kaggle Competition data) [git-ignored]
│   ├── train.csv                       # Ground-truth annotations (image_id, defect_type, mask_rle)
│   ├── sample_submission.csv           # Evaluation template (image_id, defect_type, mask_rle)
│   ├── train/                          # 11,395 raw training images (128x800 grayscale/RGB steel sheets)
│   └── test/                           # 5,000 hidden evaluation test images
│
├── models/                             # Serialized PyTorch model weights & checkpoint artifacts [git-ignored]
│   ├── best_scratch_unet.pth           # Standalone checkpoint for Model 1: ScratchUNet (31.35M params)
│   ├── scratch_unet_netron.pt          # Serialized trace for Netron architectural visualization
│   ├── segmenter/                      # Production segmenter checkpoints
│   │   └── best_scratch_unet.pth       # Multi-label checkpoint evaluated at threshold 0.50
│   ├── pretrained/                     # Transfer learning checkpoints
│   │   └── best_pretrained_resnet34.pth # 2-Phase fine-tuned ResNet-34 U-Net (24.43M params)
│   └── classifier/                     # Stage 1 Anomaly Gating models
│       └── best_classifier.pth         # Stage 1 SurfaceDefectClassifier weights (11.19M params)
│
├── notebooks/                          # Experimental workflows & competition submission notebooks
│   ├── final_notebook.ipynb            # Self-contained competition notebook (40 epochs, DDP, TTA, RLE)
│   ├── EDA.ipynb                       # Exploratory Data Analysis: class priors, RLE run-length geometry
│   ├── milestone-1.ipynb               # Milestone 1: Problem formulation, EDA, metric design, baseline
│   ├── milestone-2.ipynb               # Milestone 2: Baseline ScratchUNet architecture & training
│   ├── milestone-3.ipynb               # Milestone 3: Comparative modeling (ASPP, scSE, ResNet-34 transfer)
│   ├── milestone-5.ipynb               # Milestone 5: Calibration sweeps, 4-way TTA, ensembling & budgeting
│   ├── validation_Analysis.ipynb       # Error analysis: false positive / false negative attribution
│   └── visual_error_analysis.ipynb     # Qualitative failure-case visualizer on validation holdout
│
├── reports/                            # Comprehensive analytical documentation and exported graphics
│   ├── SurfaceGuard_AI_Final_Project_Report.md  # Official Comprehensive Final Project Report (BSDA2001P)
│   ├── milestone-1-report.md           # Milestone 1 technical milestone documentation
│   ├── figures/                        # High-resolution vector visualizations exported from W&B
│   │   ├── wandb_loss_convergence.svg  # Multi-model training and validation ComboLoss across 40 epochs
│   │   ├── wandb_official_score_progression.svg # Official competition metric progression across epochs
│   │   └── wandb_foreground_dice_progression.svg # Mean Foreground Dice dynamics over 40 epochs
│   └── validation_analysis/            # Analytical charts from holdout error inspection
│       ├── per_class_metrics_comparison.png      # Defect-wise radar & bar comparative performance
│       ├── probability_distributions_and_roc.png # Defect vs. clean sheet probability histograms & ROC
│       ├── tp_boundary_fidelity.png              # Attention gate boundary localization fidelity
│       ├── loss_attribution_pareto.png           # Pareto analysis of segmentation loss contributors
│       ├── area_distributions_log_scale.png      # Defect connected-component pixel area distributions
│       ├── gallery_high_fidelity_true_positives.png # Qualitative gallery of successful detections
│       ├── gallery_degraded_true_positives.png   # Segmentations with partial boundary degradation
│       ├── gallery_critical_false_positives.png  # False alarm cases on clean steel surfaces
│       └── gallery_critical_false_negatives.png  # Faint, missed defect instances
│
├── scripts/                            # Automation, tooling, and reporting utilities
│   └── export_wandb_figures.py         # Direct W&B GraphQL exporter for metric logs & vector figures
│
└── src/                                # Modular production Python source code
    ├── dataset.py                      # DefectDataset, DefectAwareSampler (12x rare defect oversampling)
    ├── model.py                        # ScratchUNet, ResAttentionUNet (scSE, ASPP, Attention Gates)
    ├── losses.py                       # ComboLoss (Focal Loss + Soft Dice + BCE Classification loss)
    ├── metrics.py                      # Multi-objective competition metric engine (Dice, FP Silence, Empty Acc)
    ├── train.py                        # Standardized single-GPU training loop with OneCycleLR scheduling
    ├── train_resattunet.py             # 2-GPU DistributedDataParallel (DDP) runner via torchrun (NCCL)
    ├── train_pretrained.py             # 2-Phase progressive transfer learning trainer for ResNet-34
    ├── train_classifier.py             # Dedicated Stage 1 SurfaceDefectClassifier training pipeline
    ├── inference.py                    # 4-Way Flip Test-Time Augmentation (TTA) and RLE encoding
    ├── evaluate.py                     # Standalone holdout evaluator with confusion matrix logging
    └── utils.py                        # Linux glibc memory tuning, seed setting, and RLE encode/decode
```

---

## 2. Model Zoo & Comparative Performance

We developed and benchmarked three distinct model architectures alongside a Stage 1 gating classifier:

| Architecture | Paradigm | Parameter Count | Disk Size | Best Val Score | Kaggle Public Score | Key Architectural Innovations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1: ScratchUNet** | From Scratch | $31.35\text{ M}$ | $119.6\text{ MB}$ | $0.5884$ | $0.54210$ | 4-Stage classical encoder-decoder, DoubleConv blocks, skip connections |
| **Model 2: ResAttentionUNet** | From Scratch (SOTA) | **$18.26\text{ M}$** | **$69.7\text{ MB}$** | **$0.7185$** | **$0.65905$** | Residual scSE attention, Anisotropic ASPP bottleneck, Additive Attention Gates |
| **Model 3: PretrainedResNet34** | Transfer Learning | $24.43\text{ M}$ | $93.2\text{ MB}$ | $0.6936$ | $0.61840$ | ImageNet ResNet-34 backbone, 2-phase progressive fine-tuning ($\eta_{\text{enc}} = 10^{-5}$) |
| **Stage 1: SurfaceClassifier** | Anomaly Gating | $11.19\text{ M}$ | $42.7\text{ MB}$ | $0.6817$ | — | 5-stage residual backbone with scSE pooling; consensus gating suppressor |

---

## 3. Official Competition Evaluation Metric

In industrial manufacturing, falsely flagging clean sheets causes costly unneeded assembly-line halts, while missing severe defects allows flawed metal to ship. The competition balances this via a weighted composite metric:

$$\text{Official Score} = 0.70 \times \text{ForegroundDice} + 0.20 \times \text{FalsePositiveSilenceScore} + 0.10 \times \text{EmptySheetAccuracy}$$

* **Foreground Dice ($70\%$):** Evaluated strictly on defective image-class slices ($|Y| > 0$).
* **False Positive Silence ($20\%$):** Penalizes false-alarm pixel volume on ground-truth clean sheets ($|Y| = 0$).
* **Empty Sheet Accuracy ($10\%$):** Binary classification reward when zero pixels are predicted on clean sheets ($|X| = 0$).

---

## 4. Key Engineering Innovations

1. **Two-Stage Consensus Gating:** Blends an auxiliary `SurfaceDefectClassifier` with the segmenter's internal classification head ($P_{\text{consensus}} = 0.5 \cdot P_{\text{clf}} + 0.5 \cdot P_{\text{seg\_cls}}$).
2. **Soft Quadratic Gating:** Smoothly attenuates low-confidence background predictions without threshold cliff edges:
   $$\mathbf{M}_{\text{gated}} = \mathbf{M} \cdot \min\left(1.0, \; \left(\frac{P_{\text{consensus}}}{\tau_{\text{gate}}}\right)^2\right)$$
3. **Calibrated Connected-Component Filtering:** 8-connectivity area filtering (`cv2.connectedComponentsWithStats`) eliminates small noise speckles below class thresholds ($\mathbf{A}_{\min} = [A: 60, B: 100, C: 350, D: 250]$ px).
4. **4-Way Flip Test-Time Augmentation (TTA):** Averages continuous probabilities across original, horizontal, vertical, and diagonal flips prior to thresholding.
5. **2-GPU Distributed Data Parallel (DDP):** Launched via `torchrun` with true process isolation (NCCL backend) and glibc multi-arena tuning (`MALLOC_ARENA_MAX=2`), bounding host RAM to $<4\text{ GB}$ and preventing OOM crashes (Exit Code 137).

---

## 5. Quickstart & Reproduction Guide

### Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/Dineshwarck/dl-genai-project-26-t3.git
cd dl-genai-project-26-t3

# 2. Install dependencies
pip install -r requirements.txt
```

### Export W&B Experiment Plots
```bash
# Set your Weights & Biases API Key in .env (or environment)
python scripts/export_wandb_figures.py
# Exports wandb_loss_convergence.svg and wandb_official_score_progression.svg to reports/figures/
```

### Multi-GPU Training (ResAttentionUNet via DDP)
```bash
# Launch 2-GPU Distributed Training via torchrun
torchrun --nproc_per_node=2 src/train_resattunet.py
```

### Full Standalone Pipeline Execution
Run [`notebooks/final_notebook.ipynb`](notebooks/final_notebook.ipynb) on Kaggle or a dual-GPU workstation. The notebook is completely self-contained, auto-detects GPU hardware accelerators, executes 40-epoch training, performs 4-way TTA inference, and exports the verified `submission.csv`.

---

## 6. References & Disclosures

Detailed academic paper citations, video lectures from [@tilestats](https://www.youtube.com/@tilestats), and Generative AI developer tooling disclosures (ChatGPT, Antigravity) are comprehensively documented in Section 8 of the [Final Project Report](reports/SurfaceGuard_AI_Final_Project_Report.md).

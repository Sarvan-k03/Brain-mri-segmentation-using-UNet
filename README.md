# 🧠 Tumor Segmentation using UNet with Heatmap Guidance

> **Lab 07** · Brain MRI Tumor Segmentation · 25 March 2026  
> Dataset: [LGG MRI Segmentation — Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

---

## Overview

This project implements a supervised brain tumor segmentation pipeline on the **LGG (Lower Grade Glioma) MRI dataset**. Two approaches are compared:

| Approach | Input | Model |
|---|---|---|
| **Baseline** | MRI image only | UNet (`in_ch=1`) |
| **Heatmap-Guided** | MRI image + VAE ROI heatmap | UNet (`in_ch=2`) |

Both models are trained exclusively on abnormal (tumour-positive) slices and evaluated against ground truth segmentation masks using Dice Score and IoU.

---

## Pipeline

```
LGG MRI Dataset
      │
      ├─► STEP 1 ── VAE Training on NORMAL slices
      │             └─► Reconstruction-error heatmaps for ABNORMAL slices
      │                          │
      ├─► STEP 2 ── Approach 1 (Baseline)
      │             Input : MRI only  |  Model : UNet (in_ch=1)
      │
      └─► STEP 3 ── Approach 2 (Heatmap-Guided)
                    Input : MRI + VAE Heatmap (2ch)  |  Model : UNet (in_ch=2)
```

### Why VAE for Heatmaps?

A VAE trained **only on normal (tumour-free) MRI slices** learns the distribution of healthy brain appearance. When it tries to reconstruct an **abnormal slice**, pixels with tumour tissue are *out-of-distribution* — so reconstruction error is highest exactly there. This per-pixel error map becomes the **ROI Heatmap** (no manual labels required).

| Step | What Happens |
|---|---|
| Train VAE | Only on normal slices — VAE learns the healthy brain distribution |
| Inference on abnormal | Tumour pixels are out-of-distribution → high reconstruction error |
| Heatmap | Per-pixel squared error, Gaussian-smoothed, normalised → ROI heatmap |

---

## Repository Structure

```
.
├── brain-mri-segmentation-using-unet.ipynb   # Main notebook
├── checkpoints/
│   ├── vae.pth                               # Saved VAE weights
│   ├── best_baseline.pth                     # Best baseline UNet weights
│   └── best_heatmap_guided.pth              # Best heatmap-guided UNet weights
├── roi_heatmaps/                             # VAE-generated heatmaps (auto-populated)
└── results/
    ├── vae_loss.png
    ├── vae_heatmap_samples.png
    ├── training_curves.png
    ├── approach1_samples.png
    ├── approach2_samples.png
    ├── comparison_bar.png
    └── score_distributions.png
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install torch torchvision albumentations opencv-python matplotlib scikit-learn tqdm Pillow
```

### Dataset

Download the LGG MRI Segmentation dataset from Kaggle:

```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
```

Update `Config.DATA_DIR` in the notebook to point to the dataset root:

```
kaggle_3m/
  └── <PatientID>/
        ├── <slice>.tif
        └── <slice>_mask.tif
```

---

## Configuration

All hyperparameters are centralised in the `Config` class:

```python
class Config:
    DATA_DIR       = '/path/to/lgg-mri-segmentation/kaggle_3m'
    IMG_SIZE       = 256
    
    # VAE
    VAE_EPOCHS     = 20        # raise to 40 for better heatmaps
    VAE_BATCH      = 16
    VAE_LR         = 1e-3
    VAE_LATENT_DIM = 128

    # UNet Segmentation
    SEG_EPOCHS     = 50
    SEG_BATCH      = 8
    SEG_LR         = 1e-4
    TRAIN_SPLIT    = 0.8
    BCE_WEIGHT     = 0.5
    DICE_WEIGHT    = 0.5
```

---

## Models

### VAE (Variational Autoencoder)

- **Encoder:** 5-layer strided convolutions (1→32→64→128→256→512), fully connected layers for `μ` and `log σ²`
- **Decoder:** Fully connected + 5-layer transposed convolutions back to 256×256
- **Latent dimension:** 128
- **Loss:** Reconstruction (MSE) + β-weighted KL divergence

### UNet (Segmentation)

Standard encoder-decoder architecture with skip connections:

- **Encoder:** 4 downsampling blocks (Conv → BN → ReLU → MaxPool)
- **Bottleneck:** Double convolution
- **Decoder:** 4 upsampling blocks with skip connections
- **Output:** Single-channel sigmoid mask
- **Baseline:** `in_ch=1` (MRI only)
- **Heatmap-Guided:** `in_ch=2` (MRI + VAE heatmap concatenated as 2nd channel)

---

## Loss Function

```
Combined Loss = (BCE_WEIGHT × BCE Loss) + (DICE_WEIGHT × Dice Loss)
             = 0.5 × BCE + 0.5 × Dice   [default]
```

| Component | Role |
|---|---|
| BCE | Pixel-level cross-entropy; handles class imbalance |
| Dice Loss | Directly optimises overlap; prevents all-zero prediction |
| Combined 0.5+0.5 | Balances precision (BCE) and recall-oriented overlap (Dice) |

---

## Data Augmentation

Training augmentations applied via `albumentations`:

- Horizontal & vertical flips
- Random 90° rotations
- Shift, scale, and rotate (±15°)
- Elastic transform
- Random brightness/contrast
- Normalisation

---

## Training

The notebook runs in sequence:

1. **Train VAE** on normal slices → save `vae.pth`
2. **Generate heatmaps** for all abnormal slices → save to `roi_heatmaps/`
3. **Train Baseline UNet** (image only) → save `best_baseline.pth`
4. **Train Heatmap-Guided UNet** (image + heatmap) → save `best_heatmap_guided.pth`

Optimizer: `AdamW` · Scheduler: `CosineAnnealingLR`

---

## Evaluation Metrics

- **Dice Score** — measures overlap between predicted and ground truth masks
- **IoU (Intersection over Union)** — stricter overlap metric, penalises over-segmentation

Both metrics are reported as mean ± std across the test set.

---

## Results & Visualisations

### Training Curves

Side-by-side plots of **Loss**, **Dice Score**, and **IoU Score** vs epoch for both approaches (train and validation).

### Approach 1 — Baseline (3 random test samples)

Each sample shows: Input MRI · Ground Truth Mask · Predicted Mask · Overlay with Dice & IoU scores.

### Approach 2 — Heatmap-Guided (3 random test samples)

Each sample shows: Input MRI · VAE ROI Heatmap · Ground Truth Mask · Predicted Mask · Overlay with Dice & IoU scores.

### Quantitative Comparison

| Metric | Baseline | Heatmap-Guided | Δ |
|---|---|---|---|
| Dice (mean) | ___ | ___ | ___ |
| IoU (mean) | ___ | ___ | ___ |

*Fill in after running the notebook.*

---

## Analysis

### Effect of ROI Heatmaps

- **Tumour Localisation:** The heatmap second channel steers encoder attention toward the ROI, reducing false positives in healthy tissue.
- **Boundary Accuracy:** Heatmap-guided UNet tends to produce tighter, more precise boundaries. IoU improvement typically exceeds Dice improvement since IoU penalises over-segmentation more heavily.
- **Failure Cases:** If the VAE heatmap mis-localises (VAE under-trained or tumour too small), performance can degrade versus baseline.

---

## References

- Buda, M., Saha, A., & Mazurowski, M. A. (2019). Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. *Computers in Biology and Medicine*, 109, 218–225.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv:1312.6114*.

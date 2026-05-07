# 🧠 Tumor Segmentation using UNet with Heatmap Guidance
 
> **Lab 07** · Brain MRI Tumor Segmentation · 25 March 2026  
> Dataset: [LGG MRI Segmentation — Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
 
---
 
## Overview
 
This project implements a **supervised brain tumor segmentation pipeline** on the LGG (Lower Grade Glioma) MRI dataset. Three approaches are compared side-by-side:
 
| # | Approach | Input Channels | Model |
|---|---|---|---|
| 1 | **Baseline** | 3 — RGB MRI | UNet (`in_ch=3`) |
| 2 | **VAE-Guided** | 4 — RGB MRI + VAE Heatmap | UNet (`in_ch=4`) |
| 3 | **GAN-Guided** | 4 — RGB MRI + GAN Heatmap | UNet (`in_ch=4`) |
 
Pre-generated ROI heatmaps from a VAE and GAN (Lab 06) are concatenated as an additional spatial-prior channel to guide the UNet's attention toward the tumor region. All models are trained **exclusively on abnormal (tumour-positive) slices** and evaluated on an unseen validation set using Dice Score and IoU.
 
---
 
## Pipeline
 
```
LGG MRI Dataset + Pre-generated ROI Heatmaps (VAE / GAN)
        │
        ├─► Filter: Abnormal Slices Only
        │
        ├─► APPROACH 1 ── Baseline UNet
        │                 Input : RGB MRI (3ch)
        │
        ├─► APPROACH 2 ── VAE-Guided UNet
        │                 Input : RGB MRI + VAE Heatmap (4ch)
        │
        └─► APPROACH 3 ── GAN-Guided UNet
                          Input : RGB MRI + GAN Heatmap (4ch)
```
 
---
 
## Repository Structure
 
```
.
├── Tumor_Segmentation_using_UNet.ipynb   # Main notebook
└── final_mri_outputs/                    # Dataset: per-sample folders
    └── <sample_id>/
          ├── <sample_id>_orig.jpg        # RGB MRI image
          ├── <sample_id>_mask.jpg        # Ground truth binary mask
          ├── <sample_id>_roi_vae.jpg     # VAE-generated ROI heatmap
          └── <sample_id>_roi_gan.jpg     # GAN-generated ROI heatmap
```
 
---
 
## Setup & Installation
 
### Prerequisites
 
- Python 3.8+
- CUDA-compatible GPU (recommended)
### Install Dependencies
 
```bash
pip install torch torchvision scikit-learn pillow numpy matplotlib seaborn tqdm
```
 
### Dataset
 
The dataset used is a pre-processed version of the LGG MRI Segmentation data augmented with VAE/GAN heatmaps from the previous lab. Update the `dataset_path` variable in the notebook to point to your local copy:
 
```python
dataset_path = '/path/to/final_mri_outputs'
```
 
Each sample folder must contain the four files listed in the repository structure above.
 
---
 
## Model Architecture — UNet
 
Standard UNet with a configurable input channel count, allowing it to accept either RGB-only or RGB + heatmap inputs without any structural changes.
 
```
Input (3ch or 4ch)
    │
    ├── Encoder
    │     DoubleConv → 64
    │     MaxPool + DoubleConv → 128
    │     MaxPool + DoubleConv → 256
    │     MaxPool + DoubleConv → 512  (Bottleneck)
    │
    └── Decoder (with skip connections)
          ConvTranspose2d + DoubleConv → 256
          ConvTranspose2d + DoubleConv → 128
          ConvTranspose2d + DoubleConv → 64
          Conv1x1 → 1  (output mask)
```
 
Each `DoubleConv` block: `Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU`
 
---
 
## Data Preprocessing & Augmentation
 
- Images loaded as **RGB**, resized to `256 × 256`
- Masks binarised at threshold 0.5
- Heatmaps normalised to `[0, 1]`
- ImageNet normalisation applied to the MRI channel (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
**Training augmentations** (applied consistently to image, mask, and both heatmaps):
 
| Augmentation | Probability |
|---|---|
| Horizontal flip | 50% |
| Vertical flip | 50% |
| Random 90° rotation (0°, 90°, 180°, 270°) | Always |
 
Strict 90° rotations are used to avoid blurring binary mask edges.
 
---
 
## Loss Function
 
```
Combined Loss = 0.5 × BCE Loss + 0.5 × Dice Loss
```
 
| Component | Role |
|---|---|
| **BCE Loss** | Pixel-wise accuracy; handles class imbalance |
| **Dice Loss** | Global shape/overlap optimisation; prevents all-zero predictions |
 
---
 
## Training Configuration
 
| Parameter | Value |
|---|---|
| Epochs | 60 |
| Batch size | 8 |
| Image size | 256 × 256 |
| Optimizer | Adam (`lr=1e-4`) |
| LR Scheduler | ReduceLROnPlateau (`patience=5`, `factor=0.5`) |
| Train / Val split | 80 / 20 |
| Seed | 42 |
 
---
 
## Evaluation Metrics
 
- **Dice Score** — measures overlap between predicted and ground truth masks (F1-equivalent for segmentation)
- **IoU (Intersection over Union)** — stricter overlap metric; penalises over-segmentation more heavily than Dice
---
 
## Results
 
### Final Test-Set Metrics
 
| Model | Dice Score | IoU Score |
|---|---|---|
| Baseline UNet | 0.8519 | 0.7471 |
| VAE-Guided UNet | **0.8547** | **0.7507** |
| GAN-Guided UNet | 0.8413 | 0.7320 |
 
### Visualisations Generated
 
- **Validation Loss, Dice, and IoU curves** — all three approaches overlaid per epoch
- **Visual comparison grid** — random test samples × 5 columns (Original MRI · Ground Truth · Baseline Pred · VAE Pred · GAN Pred), with Dice scores annotated per prediction
---
 
## Analysis & Key Findings
 
### Effect of ROI Heatmaps on Tumor Localization
 
The heatmap channel acts as a **soft spatial attention prior**, steering the UNet encoder toward the anomalous region. The VAE heatmap's smooth, centrally-concentrated activation confirms the tumor core for the UNet, reducing false positives in healthy tissue that shares similar MRI intensity with tumour regions. Conversely, the GAN heatmap's noisier, fragmented activations can mislead the UNet into localising tumours in entirely healthy tissue — a clear demonstration of the *garbage in, garbage out* principle.
 
### Effect of ROI Heatmaps on Boundary Accuracy
 
Upstream heatmaps are inherently blurry or fragmented at tumour edges. By combining the heatmap with the full-resolution RGB MRI, the UNet uses the heatmap to decide *where* to look and the sharp MRI pixel gradients to draw a crisp, pixel-accurate boundary — achieving better delineation than the heatmap alone could provide.
 
### Model-by-Model Observations
 
| Model | Strengths | Weaknesses |
|---|---|---|
| **Baseline** | Highly accurate; strong generalisation from RGB alone | Occasionally bloated boundaries; struggles with irregular star-shaped tumour extensions |
| **VAE-Guided** | Smoothest, most complete tumour mass capture; reliable centralised spatial prior | Modest gain; dependent on VAE heatmap quality |
| **GAN-Guided** | Can highlight subtle anomalies | Prone to false positives; noisy heatmaps actively mislead the UNet into incorrect regions |
 
### Conclusion
 
Fusing unsupervised anomaly detection heatmaps with supervised UNet segmentation produces a robust medical imaging pipeline — but the benefit is entirely contingent on heatmap quality. The smooth, continuous VAE prior yielded the best overall performance (`+0.28pp Dice`, `+0.36pp IoU` over baseline), while the artifact-prone GAN prior actively degraded performance below the unaided baseline. The downstream UNet is a highly sensitive receiver: it amplifies good guidance and equally amplifies bad guidance.
 
---
 
## References
 
- Buda, M., Saha, A., & Mazurowski, M. A. (2019). Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. *Computers in Biology and Medicine*, 109, 218–225.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv:1312.6114*.
 

# SAR2EO: Cross-Modal SAR-to-Optical Satellite Image Translation using CycleGAN

## Overview

SAR2EO is a deep learning framework for translating Synthetic Aperture Radar (SAR) imagery into Electro-Optical (EO) satellite imagery using Cycle-Consistent Generative Adversarial Networks (CycleGANs).

The project addresses the challenge of interpreting SAR imagery by generating optical-like representations from radar observations. Since SAR sensors operate independently of daylight and weather conditions while optical sensors provide more intuitive visual information, this translation framework bridges the gap between the two modalities.

The system learns mappings between SAR and EO domains without requiring perfectly aligned image pairs, making it suitable for real-world remote sensing applications.

---

## Problem Statement

Synthetic Aperture Radar imagery offers several advantages:

- All-weather imaging capability
- Day and night operation
- Penetration through clouds and atmospheric disturbances

However, SAR images are difficult for human operators to interpret due to:

- Speckle noise
- Radar-specific scattering effects
- Lack of natural visual appearance

This project aims to generate optical-like satellite imagery from SAR observations while preserving structural and geographical information.

---

## Project Objectives

- Learn SAR-to-EO image translation using CycleGAN
- Preserve spatial structures and land-cover characteristics
- Generate visually realistic EO imagery
- Evaluate translation quality using quantitative image similarity metrics
- Analyze performance across multiple EO spectral band configurations

---

# Methodology

## CycleGAN Framework

The model employs a Cycle-Consistent Generative Adversarial Network consisting of:

### Generator A → B

Transforms SAR imagery into EO imagery.

```text
SAR → Generated EO
```

### Generator B → A

Transforms EO imagery back into SAR imagery.

```text
EO → Reconstructed SAR
```

### Discriminator A

Distinguishes between:

- Real SAR images
- Generated SAR images

### Discriminator B

Distinguishes between:

- Real EO images
- Generated EO images

---

## Cycle Consistency

Cycle consistency enforces information preservation.

Forward cycle:

```text
SAR → EO → SAR
```

Backward cycle:

```text
EO → SAR → EO
```

This constraint prevents generators from producing arbitrary outputs and ensures semantic consistency between domains.

---

# Model Architecture

## Generator Network

The generator follows a residual encoder-decoder architecture.

Components:

- Initial Convolution Block
- Downsampling Layers
- Residual Blocks
- Upsampling Layers
- Output Convolution Layer

Features:

- Reflection Padding
- Instance Normalization
- Residual Learning
- ReLU Activations

The residual blocks allow the network to preserve structural information while learning cross-domain transformations.

---

## Discriminator Network

A PatchGAN discriminator is employed.

Characteristics:

- Classifies image patches rather than entire images
- Encourages high-frequency realism
- Reduces computational complexity
- Improves texture generation

---

# Dataset

## Data Source

The dataset consists of paired SAR and EO image tensors stored as:

```text
.pt
```

files.

Each sample contains:

- SAR satellite imagery
- Corresponding EO imagery

---

## Preprocessing

Several preprocessing operations are applied:

### Tensor Formatting

Images are converted to:

```text
[C, H, W]
```

format for PyTorch compatibility.

### Data Augmentation

To improve model generalization:

- Random Horizontal Flips
- Random Vertical Flips
- Random Rotations

### Normalization

Pixel values are normalized to:

```text
[-1, 1]
```

which improves GAN training stability.

---

# Experimental Design

The project investigates multiple SAR-to-EO translation settings.

## Part A

### Objective

Translate SAR imagery into a high-dimensional EO representation.

### Configuration

```text
Input:
  SAR Bands = 2

Output:
  EO Bands = 13
```

### Use Case

Full multispectral optical reconstruction.

---

## Part B

### Objective

Generate selected EO spectral bands useful for vegetation and environmental analysis.

### Configuration

```text
Input:
  SAR Bands = 2

Output:
  EO Bands = 3
```

Bands include combinations of:

- NIR
- SWIR
- Red Edge

### Use Case

Remote sensing analytics and vegetation monitoring.

---

## Part C

### Objective

Enhanced multispectral reconstruction with refined training and evaluation procedures.

### Configuration

```text
Input:
  SAR Bands = 2

Output:
  EO Bands = Multiple Optical Bands
```

### Enhancements

- Improved training pipeline
- Checkpointing system
- Visualization generation
- Automated metric computation

---

# Loss Functions

The training objective combines multiple losses.

## Adversarial Loss

Encourages generators to produce realistic outputs capable of fooling discriminators.

```text
L_GAN
```

---

## Cycle Consistency Loss

Preserves information across domain translation.

```text
L_cycle
```

---

## Identity Loss

Helps preserve color and spectral consistency.

```text
L_identity
```

---

## Final Objective

```text
L_total =
L_GAN
+ λ_cycle L_cycle
+ λ_identity L_identity
```

---

# Training Pipeline

## Optimizer

Adam Optimizer

```text
Learning Rate = 2e-4
β1 = 0.5
β2 = 0.999
```

---

## Checkpointing

The training pipeline automatically saves:

- Generator weights
- Discriminator weights
- Optimizer states
- Current epoch

This allows interrupted training sessions to resume seamlessly.

---

## Monitoring

Training progress is monitored using:

- Generator Loss
- Discriminator Loss
- Cycle Consistency Loss
- Identity Loss

Progress bars are displayed using:

```text
tqdm
```

---

# Evaluation Metrics

The generated EO imagery is evaluated using:

## PSNR

Peak Signal-to-Noise Ratio

Measures reconstruction fidelity.

Higher values indicate better image quality.

---

## SSIM

Structural Similarity Index

Measures structural consistency between generated and ground-truth imagery.

Higher values indicate stronger perceptual similarity.

---

# Output Artifacts

The repository contains two dedicated folders:

## metrics/

Stores:

- Evaluation CSV files
- PSNR scores
- SSIM scores
- Experiment summaries

Example:

```text
metrics/
├── metrics_parta.csv
├── metrics_partb.csv
└── metrics_partc.csv
```

---

## generated_images/

Stores:

- Generated EO outputs
- SAR inputs
- Ground-truth EO images
- Visualization comparisons

Example:

```text
generated_images/
├── sample_01.png
├── sample_02.png
├── sample_03.png
└── ...
```

---

# Repository Structure

```text
SAR2EO/
│
├── sar-2-eo-parta.ipynb
├── sar2eo-part-b.ipynb
├── sar2eo-part-c.ipynb
│
├── generated_images/
│
├── metrics/
│
├── README.md
│
└── checkpoints/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/SAR2EO.git

cd SAR2EO
```

---

## Install Dependencies

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-image pandas
```

---

# Running the Project

Open the notebook corresponding to the experiment:

### Part A

```text
sar-2-eo-parta.ipynb
```

### Part B

```text
sar2eo-part-b.ipynb
```

### Part C

```text
sar2eo-part-c.ipynb
```

Run all cells sequentially.

The notebooks will:

1. Load SAR and EO datasets
2. Perform preprocessing
3. Train the CycleGAN model
4. Generate EO predictions
5. Compute evaluation metrics
6. Save visual outputs

---

# Applications

Potential applications include:

- Cloud-free satellite visualization
- Disaster monitoring
- Agricultural analysis
- Environmental monitoring
- Land-cover mapping
- Change detection
- Remote sensing interpretation
- Military and intelligence analysis

---

# Technologies Used

- Python
- PyTorch
- CycleGAN
- NumPy
- Matplotlib
- Scikit-Image
- Pandas
- tqdm

---

# Future Work

Potential improvements include:

- Attention-guided CycleGANs
- Multi-scale discriminators
- Spectral consistency losses
- Transformer-based generators
- Diffusion-based SAR-to-EO translation
- Full Sentinel-1 to Sentinel-2 reconstruction
- Quantitative downstream task evaluation

---

# Author

**Shyla Vijay**

SAR-to-Optical Image Translation using CycleGAN for Remote Sensing Applications.

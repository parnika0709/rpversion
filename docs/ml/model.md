# ML Model Architecture: 2D Attention U-Net

## Overview

The current baseline model is a **2D Attention U-Net**. It is designed to be lightweight and modular, allowing for future replacement with more complex 3D architectures (e.g., nnU-Net) without breaking the API.

## Architecture Details

- **Input**: 3-channel slices (T1ce, T2, FLAIR).
- **Output**: 4-channel segmentation map (Background, WT, TC, ET).
- **Attention Gates**: Integrated into the decoder path to focus on relevant tumor regions and suppress irrelevant background.
- **Implementation**: Built using the `monai.networks.nets.AttentionUnet` implementation.

## Training Workflow

1. **Data Loading**: Uses `BratsDataset` to load NIfTI volumes.
2. **Preprocessing**: Slices are extracted and preprocessed via `train_transforms`.
3. **Optimization**: AdamW optimizer with a OneCycleLR scheduler.
4. **Loss Function**: Combined Dice + Cross Entropy Loss.
5. **Validation**: Evaluated on a separate patient split using Dice and IoU metrics.

## Evaluation Workflow

Evaluation is performed on the test split:
1. **Inference**: Model produces raw logits.
2. **Postprocessing**: Masks are converted back to BraTS-native labels {0, 1, 2, 4}.
3. **Metrics**: Compute Dice, IoU, and Hausdorff distance for each region (WT, TC, ET).

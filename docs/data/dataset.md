# Dataset Requirements and Setup

## BraTS Directory Structure

The system expects the BraTS dataset to be organized as follows:

```
data/raw/
└── BraTS2021_Training/
    ├── BraTS2021_00001/
    │   ├── BraTS2021_00001_t1.nii.gz
    │   ├── BraTS2021_00001_t1ce.nii.gz
    │   ├── BraTS2021_00001_t2.nii.gz
    │   ├── BraTS2021_00001_flair.nii.gz
    │   └── BraTS2021_00001_seg.nii.gz
    └── ...
```

## Setup Instructions

1. **Data Placement**: Place your raw BraTS data in `data/raw/`.
2. **Configuration**: Set the environment variable `BT_DATA_ROOT` to point to this directory.
3. **Splitting**: The system automatically generates a patient-level split saved to `data/splits/patients_split.json` to prevent data leakage.

## Preprocessing Pipeline

The pipeline uses MONAI transforms:
- **Orientation**: All volumes are oriented to 'RAS'.
- **Intensity**: Volumes are scaled using 1st and 99th percentiles and then z-score normalized based on non-zero voxels.
- **Training Augmentation**: Includes random flips, rotations, Gaussian noise, and intensity shifts.
- **Validation**: Deterministic scaling and normalization only.

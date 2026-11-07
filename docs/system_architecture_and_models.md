# NeuroRisk Lens — System Architecture, Models & Methodology

> **CLINICAL DISCLAIMER**: NeuroRisk Lens is a research and educational prototype. It is **NOT** a medical device, is **NOT** cleared or approved by any regulatory authority (e.g., FDA, EMA), and is **NOT** validated for clinical diagnosis, real-world surgical planning, or clinical decision-making. Anatomical proximity analyses, overlap percentages, and surgical risk tiers are illustrative and heuristic. Always consult a qualified neurosurgeon, neuroradiologist, or medical multidisciplinary team.

---

## 1. Overall System Architecture

The NeuroRisk Lens platform is structured as a decoupled monorepo composed of three principal tiers:

```
┌────────────────────────────────────────────────────────┐
│             Presentation Layer (apps/api/static)       │
│  - Interactive Web Dashboard (Three.js / HTML5)        │
│  - Multi-planar 2D slice viewer (Axial/Coronal/Sag)    │
│  - 3D Iso-surface volume rendering                     │
│  - Structured Risk Assessment & Plain-Language Reports │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP REST / JSON
┌───────────────────────────▼────────────────────────────┐
│             Application Services (apps/api)            │
│  - FastAPI REST API (Endpoints: /cases, /predict)       │
│  - SQLite Database (Case metadata & prediction logs)   │
│  - Upload & Storage Lifecycle Management               │
│  - Inference Dispatcher via Predictor Registry         │
└───────────────────────────┬────────────────────────────┘
                            │ Python API / In-Memory Tensors
┌───────────────────────────▼────────────────────────────┐
│             Machine Learning & Risk Core               │
│                  (packages/ml)                         │
│  ┌───────────────────────┐  ┌───────────────────────┐  │
│  │ 2D Attention U-Net    │  │ 3D Volumetric U-Net   │  │
│  │ (Permanent Baseline)  │  │ (Independent Model)   │  │
│  └───────────┬───────────┘  └───────────┬───────────┘  │
│              └─────────────┬────────────┘              │
│                            ▼                           │
│  - Standardized Ingestion & Validation Preprocessing   │
│  - Post-Processing & Connected Components Filtering   │
│  - Harvard-Oxford MNI152 Standard-Space Atlas Mapping   │
│  - Surface-to-Surface Distance & Dual Overlap Metrics  │
│  - Explainable Rule-Based & Verifiable Report Engine   │
└────────────────────────────────────────────────────────┘
```

The system uses a strict **`Predictor` interface contract** (`braintumor.registry.Predictor`). Both segmentation models implement `predict(VolumeInput) -> PredictionOutput`, allowing models to be evaluated independently without changing backend routes, database schemas, or frontend interfaces.

---

## 2. Segmentation Models

### 2.1 Baseline 2D Attention U-Net (Permanent Baseline)

- **Role**: Permanent baseline model and default operational model for the application.
- **Architecture**: 5-level encoder-decoder with additive spatial attention gates (`monai.networks.nets.AttentionUnet`).
- **Input Dimensions**: 3-channel axial slices: $(B, 3, H, W)$ corresponding to co-registered modalities:
  1. $0 = \text{T1ce}$ (T1-weighted contrast-enhanced)
  2. $1 = \text{T2}$ (T2-weighted)
  3. $2 = \text{FLAIR}$ (Fluid-Attenuated Inversion Recovery)
- **Output Labels**: 4 classes: $\{0: \text{Background}, 1: \text{NCR/NET}, 2: \text{EDEMA}, 3: \text{ET}\}$, converted to native BraTS labels $\{0, 1, 2, 4\}$.
- **Parameters**: 7,940,420 total trainable parameters.
- **Slice Sampling Strategy**: During training, extracts 8 slices per volume with a 70% foreground bias to counter class imbalance.
- **Full Volume Inference**: Processes axial slices sequentially in batches (chunk size 16) and reassembles the 3D volume, with optional Monte Carlo dropout for epistemic uncertainty estimation.

### 2.2 Independent Volumetric 3D U-Net (Experimental Model)

- **Role**: Independent volumetric model developed to provide authentic 3D spatial context and address multi-planar continuity.
- **Architecture**: Volumetric 3D encoder-decoder with genuine 3D convolutions (`spatial_dims=3`), residual units (`num_res_units=2`), and skip connections (`monai.networks.nets.UNet`).
- **Input Dimensions**: 3D volumetric tensor: $(B, 3, D, H, W)$ with the same three MRI modalities.
- **Channels & Strides**: Channels $(16, 32, 64, 128)$ with 3D strides $(2, 2, 2)$, totaling 1,190,549 parameters.
- **Volumetric Inference**: Employs Gaussian-blended sliding-window volumetric inference (`monai.inferers.sliding_window_inference`) with configurable region-of-interest patches (e.g., $64 \times 64 \times 64$), preventing memory overflow on constrained devices.
- **Independence Guarantee**: Has separate source files (`braintumor/models/unet_3d.py`), separate training script (`scripts/train_3d.py`), separate configuration, and independent checkpoint directories (`runs/exp_3d/`).

### 2.3 Rationale for Retaining Both Models

| Attribute | 2D Attention U-Net | 3D Volumetric U-Net | Rationale for Retention |
| :--- | :--- | :--- | :--- |
| **Status** | Permanent Default Baseline | Independent Experimental Model | Preserves tested operational stability while enabling 3D research |
| **Parameters** | 7,940,420 | 1,190,549 | 3D uses compact channel depth suitable for 8 GB RAM |
| **Memory Footprint** | ~110 MB Peak RAM | ~104 MB Peak RAM | Both run safely on M1 Mac with 8 GB unified memory |
| **Inference Latency** | ~9.0 - 12.5 s (CPU, 128³ vol) | ~3.1 - 4.3 s (CPU, 128³ vol) | 3D sliding window with large stride yields rapid volumetric passes |
| **Spatial Continuity** | High in-plane; slice boundary effects | Fully isotropic 3D continuity | 3D eliminates step artifacts between axial slices |
| **Training State** | Fully trained baseline weights available | Smoke-tested & pipeline-verified; full convergence pending GPU training | 2D provides reliable segmentation today; 3D provides extensible foundation |

---

## 3. Data Processing & Dataset Pipeline

### 3.1 BraTS Multi-Modal Ingestion

The repository processes BraTS multi-sequence NIfTI collections containing:
- `*_t1ce.nii.gz` / `*_t1ce.nii`
- `*_t2.nii.gz` / `*_t2.nii`
- `*_flair.nii.gz` / `*_flair.nii`
- `*_seg.nii.gz` / `*_seg.nii` (training/evaluation ground truth)

### 3.2 Deterministic Validation Preprocessing

To ensure reproducibility across training, inference, and evaluation, validation preprocessing follows strict deterministic transforms:
1. **Reorientation**: Reoriented to standard radiological coordinate space (`Orientationd(axcodes="RAS")`).
2. **Resampling**: Isotropic resampling to $1.0\,\text{mm} \times 1.0\,\text{mm} \times 1.0\,\text{mm}$ voxel spacing via bilinear interpolation for intensities and nearest-neighbor for labels (`Spacingd`).
3. **Intensity Clipping**: Robust intensity scaling between the 1st and 99th percentiles (`ScaleIntensityRangePercentilesd(lower=1.0, upper=99.0)`).
4. **Z-Score Normalization**: Channel-wise zero-mean, unit-variance normalization over non-zero brain voxels (`NormalizeIntensityd(nonzero=True, channel_wise=True)`).

### 3.3 Patient-Level Splitting (Leakage Prevention)

To prevent data leakage, splitting is strictly enforced at the **patient level** via `discover_and_split()`:
- No patient ID can appear in both training and validation/testing splits.
- Formally verified via `assert_no_patient_overlap()`.
- Default deterministic split ratio: 70% Train, 15% Validation, 15% Test.

---

## 4. Harvard-Oxford Standard-Space Atlas & Surgical Risk Mapping

### 4.1 Atlas Assets & Coordinate Transformations

Rather than relying on ungrounded bounding boxes or synthetic ellipsoids, the system registers patient MRI volumes directly to the official **Harvard-Oxford Cortical and Subcortical Structural Atlases** in standard MNI152 2 mm space.

Physical files embedded and preserved:
- `HarvardOxford-cort-maxprob-thr25-2mm.nii.gz` (Cortical structural atlas)
- `HarvardOxford-sub-maxprob-thr25-2mm.nii.gz` (Subcortical structural atlas)
- `HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz` (Lateralized cortical atlas)
- Corresponding XML label definitions (`HarvardOxford-Cortical.xml`, `HarvardOxford-Subcortical.xml`)

### 4.2 Mathematical Transformation

Patient MRI space coordinates $x_{\text{mri}}$ are mapped to atlas MNI152 space $x_{\text{atlas}}$ via the composite affine transformation matrix $M$:

$$M = A_{\text{atlas}}^{-1} \cdot A_{\text{mri}}$$

Where $A_{\text{mri}}$ is the patient volume affine and $A_{\text{atlas}}$ is the Harvard-Oxford MNI152 affine. Binary segmentation masks are mapped into standard space using nearest-neighbor push-forward resampling (`resample_mask_to_atlas`).

### 4.3 Spatial Quantitation & Distance Calculations

For each anatomical critical structure $R$ and segmented tumor mask $T$:
1. **Tumor Coverage (%)**: Percentage of the total tumor volume residing inside structure $R$:
   $$\text{Coverage}(T, R) = \frac{|T \cap R|}{|T|} \times 100\%$$
2. **Region Infiltration (%)**: Percentage of anatomical structure $R$ invaded by tumor $T$:
   $$\text{Infiltration}(T, R) = \frac{|T \cap R|}{|R|} \times 100\%$$
3. **Surface-to-Surface Distance (mm)**: The exact minimum Euclidean physical distance from the tumor surface to the region boundary:
   $$d_{\text{min}}(T, R) = \min_{p \in T, q \in R} \|p - q\|_2$$
   Computed using the Euclidean Distance Transform (EDT) sampled with true physical voxel spacing, with an exact boundary coordinate morphological fallback (`surface_to_surface_distance_mm`).

### 4.4 Tri-Tier Risk Stratification

The system computes heuristic risk categories:
- **High Risk**: Direct tumor overlap $\ge 5\%$ or physical distance $< 5.0\,\text{mm}$ to critical functional structures (e.g., motor cortex, Broca's area, Wernicke's area, brainstem).
- **Moderate Risk**: Overlap $> 0\%$ or physical distance between $5.0\,\text{mm}$ and $15.0\,\text{mm}$.
- **Low Risk**: Clear separation with physical distance $\ge 15.0\,\text{mm}$.
- **Negative Case Handling**: Cases with no segmented tumor are assigned `NO_TUMOR_DETECTED`, bypassing functional region proximity scores and emitting clear non-clearance warnings.

---

## 5. Evaluation Methodology & Metric Battery

The evaluation suite calculates full quantitative performance across both composite BraTS regions and individual sub-compartments:

1. **Composite Regions**:
   - **Whole Tumor (WT)**: Labels $\{1, 2, 4\}$
   - **Tumor Core (TC)**: Labels $\{1, 4\}$
   - **Enhancing Tumor (ET)**: Label $\{4\}$
2. **Sub-Compartment Classes**:
   - **NCR/NET**: Label $\{1\}$ (Necrotic / Non-enhancing core)
   - **EDEMA**: Label $\{2\}$ (Peritumoral edema)
   - **ET**: Label $\{4\}$ (Active enhancing tumor)
3. **Metric Formulations**:
   - **Dice Similarity Coefficient (DSC)**: $\frac{2|P \cap T|}{|P| + |T|}$
   - **Intersection over Union (IoU / Jaccard)**: $\frac{|P \cap T|}{|P \cup T|}$
   - **Sensitivity / Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
   - **Specificity**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$
   - **Precision (PPV)**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
   - **95th Percentile Hausdorff Distance ($HD_{95}$ in mm)**: Maximum distance between boundary surfaces at the 95th percentile, evaluated in true physical millimeter coordinates.

---

## 6. Hardware Constraints & Operational Profile

- **Host Architecture**: Apple Silicon (M1, 8-core CPU) with 8 GB unified memory.
- **Memory Management**:
  - Full-volume 3D patch extraction at large resolutions would exceed the 8 GB unified memory budget.
  - Volumetric training uses compact cubic patches ($32^3$ or $64^3$) with batch size 1 or 2.
  - Sliding-window inference evaluates overlapping sub-volumes sequentially, capping peak RAM below 120 MB.
- **Compute Compatibility**: Automatic fallback hierarchy: `CUDA -> MPS -> CPU`. On macOS environments without CUDA, training and inference execute seamlessly on CPU/MPS.
- **Recommended Full-Scale Training**: For full 50-epoch convergence on the 1,251-patient BraTS dataset, execution on an NVIDIA GPU (e.g., A100/V100 or Google Colab T4/A100) is recommended.

---

## 7. Execution Commands

### 7.1 Running Automated Tests
```bash
# Execute complete unit, integration, and risk mapping test battery (42 tests)
python3 -m pytest
```

### 7.2 Training Models
```bash
# 2D Attention U-Net Baseline Training (CPU / Colab / GPU)
python3 scripts/train.py --epochs 20 --batch_size 4

# 3D Volumetric U-Net Independent Training
python3 scripts/train_3d.py --epochs 10 --batch_size 2 --patch_size 64

# 3D Synthetic Smoke-Test (no external dataset required)
python3 scripts/train_3d.py --synthetic --epochs 2 --batch_size 1 --patch_size 32
```

### 7.3 Benchmarking & Performance Comparison
```bash
# Run multi-stage benchmark comparing 2D and 3D pipelines
python3 scripts/benchmark_inference.py --model both --n_cases 2
```

### 7.4 Comprehensive Evaluation on Patient MRI
```bash
# Evaluate baseline 2D Attention U-Net on a test patient
python3 scripts/evaluate_comprehensive.py --model attn_unet_2d --checkpoint runs/exp/best.pt --case_dir /path/to/case --output_dir outputs/evaluation

# Evaluate experimental 3D U-Net on the same patient
python3 scripts/evaluate_comprehensive.py --model unet_3d --checkpoint runs/exp_3d/best.pt --case_dir /path/to/case --output_dir outputs/evaluation_3d
```

### 7.5 Launching Backend API & Frontend Dashboard
```bash
# Launch FastAPI backend with default 2D Attention U-Net
uvicorn app.main:app --app-dir apps/api --port 8000 --reload
# Access web dashboard at http://127.0.0.1:8000/
```

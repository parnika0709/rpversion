# Technical Novelty and Engineering Contributions of NeuroRisk Lens

## 1. Context and Framing
Reviewer 1 correctly noted the importance of clearly distinguishing between:
1. Claiming an entirely new neural network architectural primitive (e.g. creating a new attention mechanism), and
2. Developing a unified, scientifically grounded computational pipeline that bridges deep multi-modal image segmentation with registered standard-space anatomical risk mapping and verifiable clinical communication.

NeuroRisk Lens does **not** claim to invent a novel convolutional kernel or custom attention block. The segmentation backbones utilize established, high-performing architectures: the baseline **2D Attention U-Net** (`monai.networks.nets.AttentionUnet`) and the experimental volumetric **3D U-Net** (`monai.networks.nets.UNet`).

Rather, the technical contribution of NeuroRisk Lens lies in the **rigorous mathematical and algorithmic synthesis of five interconnected components into a single, automated, reproducible neurosurgical risk-stratification system**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEURORISK LENS PIPELINE                           │
├──────────────────┬─────────────────┬───────────────────┬────────────────────┤
│ 1. Segmentation  │ 2. Spatial Reg. │ 3. Metric Engine  │ 4. Verifiable Rep. │
│ 2D Attn U-Net /  │ RAS -> MNI152   │ Overlap % (Tumor) │ Deterministic Num. │
│ 3D Volumetric    │ Harvard-Oxford  │ Infiltration %    │ Verifier Engine    │
│ Post-processing  │ Discrete Labels │ Surface Dist (mm) │ Plain-Language LLM │
└──────────────────┴─────────────────┴───────────────────┴────────────────────┘
```

---

## 2. Component-by-Component Contributions

### Component A: Multi-Modal MRI Segmentation Engine
- **Standardized Multi-Modal Ingestion**: Handles core BraTS modalities (T1ce, T2, FLAIR) with deterministic spatial resampling (1.0 mm isotropic), RAS orientation enforcement, and intensity scaling (1st–99th percentiles with non-zero z-score normalization).
- **Dual Independent Model Strategy**:
  - **2D Attention U-Net (Baseline & Permanent Default)**: Slices volumes along the axial axis, processes them via attention gates to prioritize non-background tumor regions, and recombines them into full 3D volumes.
  - **3D Volumetric U-Net (Independent Experimental Model)**: Uses true 3D spatial convolutions with sliding-window patch inference (`roi_size=(64,64,64)`), eliminating inter-slice boundary discontinuities while maintaining bounded VRAM consumption.
- **Class-Imbalance-Aware Loss**: Provides both standard `DiceCELoss` and `ClassWeightedDiceCELoss` designed specifically for severe volumetric imbalances between the small enhancing tumor core (ET, ~1-5% of tumor volume) and large peritumoral edema (ED).

### Component B: Standard-Space Atlas Localization (Harvard-Oxford MNI152)
- Replaces naive bounding-box approximations with actual gyral and subcortical parcel labels from the **Harvard-Oxford Probabilistic Atlas (25% threshold, 2mm isotropic MNI152 space)**.
- Enforces rigorous coordinate transformations:
  $$v_{patient} = \left(A_{patient}^{-1} \cdot M_{reg}^{-1} \cdot A_{atlas}\right) \cdot v_{atlas}$$
- Uses **nearest-neighbor interpolation (`order=0`)** for resampling categorical masks to prevent creating false intermediate label values along structural boundaries.
- Distinguishes cortical lateralization (e.g. Left vs. Right Precentral Gyrus / Motor Cortex) to preserve functional laterality (e.g., Broca's area on the dominant left hemisphere).

### Component C: Dual Overlap and Physical Proximity Metrics
Unlike tools that compute only a single binary intersection, NeuroRisk Lens maintains two distinct, clinically meaningful volume percentages:
1. **Tumor Coverage (%)**: Proportion of the patient's whole tumor volume located inside the anatomical region:
   $$\text{Coverage}_{\text{tumor}} = \frac{|\mathcal{V}_{\text{tumor}} \cap \mathcal{V}_{\text{region}}|}{|\mathcal{V}_{\text{tumor}}|} \times 100$$
2. **Region Infiltration (%)**: Proportion of the anatomical structure invaded by the tumor:
   $$\text{Infiltration}_{\text{region}} = \frac{|\mathcal{V}_{\text{tumor}} \cap \mathcal{V}_{\text{region}}|}{|\mathcal{V}_{\text{region}}|} \times 100$$
3. **Physical Surface-to-Surface Proximity**:
   Computes the minimum Euclidean distance in true millimeters ($mm$) using the physical voxel spacing via the Euclidean Distance Transform (EDT) and boundary coordinate analysis:
   $$d_{\text{min}}(\mathcal{V}_{\text{tumor}}, \mathcal{V}_{\text{region}}) = \min_{x \in \partial\mathcal{V}_{\text{tumor}}, y \in \partial\mathcal{V}_{\text{region}}} \|x - y\|_2$$

### Component D: Tri-Tier Risk Stratification with Explicit Statuses
- Evaluates functional risk based on configurable cut-offs:
  - **High Risk**: Tumor directly infiltrates the eloquent region ($\ge 5\%$ tumor overlap or $\ge 5\%$ region infiltration).
  - **Moderate Risk**: Minor direct overlap ($< 5\%$) or close proximity ($\le 10\text{ mm}$).
  - **Low Risk**: No direct overlap ($0\%$) and distance $> 10\text{ mm}$.
- **Explicit Negative Handling**:
  - Empty predicted segmentation mask outputs an explicit status: `NO_TUMOR_DETECTED` with region status `"No tumor detected"` (preventing dangerous false reassurance from labeling all regions as "Low").
  - Registration failures output `FAILED_REGISTRATION` without risk scoring.

### Component E: Deterministic-First Verifiable Plain-Language Reporting
- Strict separation between the **deterministic computation engine** (immutable numerical ground truth) and the **natural language explanation**.
- The verification engine (`verify_report_numbers`) acts as an automated safety barrier:
  - Guarantees every number in the text matches computed data.
  - Verifies that risk tiers stated for regions are not altered or hallucinated.
  - Enforces mandatory clinical disclaimers.
  - Rejects clinical directive keywords ("recommend surgery", "safe to operate", "guarantee").

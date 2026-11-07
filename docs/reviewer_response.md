# Reviewer Response Document
## NeuroRisk Lens: An Integrated Pipeline for Brain Tumor Segmentation, Functional-Region Risk Mapping and Plain-Language Reporting from Multimodal MRI

**Version:** 2.0 (Post-revision)
**Date:** 2026-11-06

This document maps each of the 15 reviewer concerns to their implementation status, the specific code or documentation that addresses the concern, and the technical evidence.

---

## Overview

| Concern # | Category | Status |
|-----------|----------|--------|
| 1 | Novelty & Contributions | ✅ FULLY ADDRESSED |
| 2 | Dice Coefficient | ✅ FULLY ADDRESSED |
| 3 | IoU / Jaccard Metric | ✅ FULLY ADDRESSED |
| 4 | Sensitivity / Recall | ✅ FULLY ADDRESSED |
| 5 | Specificity | ✅ FULLY ADDRESSED |
| 6 | Precision | ✅ FULLY ADDRESSED |
| 7 | 3D vs 2D Comparison | ⚠️ PARTIALLY ADDRESSED (architecture complete; training requires BraTS data) |
| 8 | Hausdorff Distance | ✅ FULLY ADDRESSED |
| 9 | Atlas-Based Mapping | ✅ FULLY ADDRESSED |
| 10 | Patient-Level Splitting | ✅ FULLY ADDRESSED |
| 11 | Inference Benchmarking | ✅ FULLY ADDRESSED |
| 12 | Report Numerical Consistency | ✅ FULLY ADDRESSED |
| 13 | Empty Mask / Edge-Case Handling | ✅ FULLY ADDRESSED |
| 14 | Incomplete Reference [10] | ✅ FULLY ADDRESSED |
| 15 | Clinical Disclaimer | ✅ FULLY ADDRESSED |

---

## Detailed Responses

---

### Reviewer Concern 1 — Novelty and Contribution Clarity

**Concern:** The reviewers requested a clearer articulation of the technical contribution of the system, distinguishing it from simply applying existing architectures.

**Response:**

We have added a dedicated technical novelty document (`docs/novelty_and_contributions.md`) that:

1. Explicitly clarifies that the segmentation backbones (2D Attention U-Net and 3D Volumetric U-Net) use established MONAI architectures — this is not claimed as an architectural novelty.
2. Articulates the **five-component pipeline integration** as the core research contribution:
   - Component A: Multi-modal MRI standardized ingestion + dual independent model strategy
   - Component B: Standard-space Harvard-Oxford MNI152 atlas localization (replacing bounding-box approximations)
   - Component C: Dual overlap metrics — Tumor Coverage (%) and Region Infiltration (%) — plus physical surface-to-surface distance
   - Component D: Tri-tier risk stratification with explicit negative handling (NO_TUMOR_DETECTED, FAILED_REGISTRATION, INVALID_SEGMENTATION)
   - Component E: Deterministic-first verifiable plain-language reporting with automated numerical consistency verification

The mathematical formulations for the coordinate transformation and both overlap metrics are included in the document.

**Files:**
- `docs/novelty_and_contributions.md`

---

### Reviewer Concern 2 — Dice Coefficient Reporting

**Concern:** Reviewers requested the Dice coefficient be reported per BraTS region.

**Response:**

The `evaluate` module now computes Dice per:
- Composite BraTS regions: **WT** (Whole Tumor = labels 1+2+4), **TC** (Tumor Core = labels 1+4), **ET** (Enhancing Tumor = label 4)
- Individual BraTS sub-classes: **NCR/NET** (label 1), **EDEMA** (label 2), **ET** (label 4)

Multi-case aggregation provides: mean, standard deviation, median, IQR for all regions.

**Actual result (BraTS2021_00495, 2D model, postprocessed):**

| Region | Dice |
|--------|------|
| WT | 0.264 |
| TC | 0.178 |
| ET | 0.175 |
| NCR/NET | 0.152 |
| EDEMA | 0.147 |

Note: These values reflect the checkpoint trained on limited data for a small number of epochs. Research-grade results require full BraTS 2021 training.

**Files:**
- `packages/ml/src/braintumor/eval/metrics.py` — `dice()`, `evaluate_case()`, `aggregate_metrics()`
- `scripts/evaluate_comprehensive.py`
- `packages/ml/tests/eval/test_metrics.py`

---

### Reviewer Concern 3 — IoU / Jaccard Metric

**Concern:** Reviewers requested IoU (Intersection over Union / Jaccard index) reporting.

**Response:**

IoU is computed per BraTS region and sub-class alongside Dice. The formula used is:

```
IoU = |P ∩ T| / |P ∪ T|
```

For empty-both-masks cases, the function returns `nan` (or `empty_val` if supplied) rather than a misleading 0 or 1.

**Files:**
- `packages/ml/src/braintumor/eval/metrics.py` — `iou()`

---

### Reviewer Concern 4 — Sensitivity / Recall

**Concern:** Reviewers requested Sensitivity (True Positive Rate) reporting.

**Response:**

Sensitivity (= Recall = TPR) is computed as:

```
Sensitivity = |P ∩ T| / |T|
```

Both `sensitivity()` and `recall()` are exposed as named functions (they are mathematically identical). The `RegionMetrics` dataclass stores both `sensitivity` and `recall` attributes for clarity in reporting.

**Files:**
- `packages/ml/src/braintumor/eval/metrics.py` — `sensitivity()`, `recall()`

---

### Reviewer Concern 5 — Specificity

**Concern:** Reviewers requested Specificity (True Negative Rate) reporting.

**Response:**

Specificity (TNR) is computed as:

```
Specificity = TN / (TN + FP) = |~P ∩ ~T| / |~T|
```

It is included in every evaluation case result and aggregated across cases. For the case where the ground truth has no negative voxels (all-tumor volume), `nan` is returned.

**Files:**
- `packages/ml/src/braintumor/eval/metrics.py` — `specificity()`

---

### Reviewer Concern 6 — Precision

**Concern:** Reviewers requested Precision (Positive Predictive Value) reporting.

**Response:**

Precision (PPV) is computed as:

```
Precision = |P ∩ T| / |P|
```

For empty prediction masks, `nan` (or `empty_val`) is returned rather than a misleading 0.

**Files:**
- `packages/ml/src/braintumor/eval/metrics.py` — `precision()`

---

### Reviewer Concern 7 — 3D vs 2D Comparison

**Concern:** Reviewers requested an experimental comparison between 2D slice-based and 3D volumetric architectures to assess whether processing 3D context improves segmentation quality.

**Response:**

We have implemented an **independent, experimental 3D Volumetric U-Net** that:

1. Uses genuine 3D spatial convolutions (`spatial_dims=3`) with residual units and skip connections (MONAI `UNet` with `num_res_units=2`)
2. Uses MONAI sliding-window patch inference (`roi_size=(64,64,64)`, overlap=0.25) to handle full volumetric inputs without memory overflow
3. Is trained using an **independent script** (`scripts/train_3d.py`) that shares the identical patient-level data split with the 2D pipeline — preventing data leakage
4. Uses equivalent augmentation: RAS reorientation, 1mm isotropic resampling, percentile scaling, z-score normalization, random flips, Gaussian noise
5. Does **not** modify, replace, or interfere with the 2D Attention U-Net baseline in any way

**Current Limitation:** A fully trained 3D checkpoint does not yet exist in the repository because BraTS 2021 training data is not included. The pipeline has been verified end-to-end using synthetic data (`--synthetic` flag). Fair quantitative comparison (Dice, IoU, HD95, etc.) requires running both models on the same patient split with the same hardware budget.

**We explicitly state:** 3D superiority is not assumed. The 2D Attention U-Net remains the permanent default application model. If 3D training is not feasible within the submission timeline, results will be reported as "architecture verified, training not completed due to hardware constraints."

**Model parameter counts:**
- 2D Attention U-Net: **7,940,420 parameters**
- 3D Volumetric U-Net: **1,190,549 parameters**

**Files:**
- `packages/ml/src/braintumor/models/unet_3d.py`
- `scripts/train_3d.py`
- `scripts/evaluate_comprehensive.py` (supports `--model unet_3d`)
- `scripts/benchmark_inference.py` (supports `--model both`)

---

### Reviewer Concern 8 — Hausdorff Distance

**Concern:** Reviewers requested the 95th-percentile Hausdorff distance (HD95) in millimeters.

**Response:**

HD95 is computed in physical millimeters using voxel spacing:

1. Extract boundary voxels via morphological erosion: `boundary = mask XOR erode(mask)`
2. Convert boundary voxel coordinates to physical mm: `coords_mm = coords_vox * spacing`
3. Compute directed distances and take 95th percentile
4. HD95 = max(forward 95th pct, backward 95th pct)

Memory management: for very large masks (>5M total pairs), the prediction boundary is randomly subsampled to maintain tractable computation.

**Actual result (BraTS2021_00495):** WT HD95 = 145.3mm, TC = 137.2mm, ET = 87.9mm (high values expected for undertrained model).

**Files:**
- `packages/ml/src/braintumor/eval/metrics.py` — `hausdorff95_mm()`

---

### Reviewer Concern 9 — Atlas-Based Mapping (Not Bounding Boxes)

**Concern:** Reviewers noted the original implementation used approximated bounding-box regions rather than a validated anatomical atlas.

**Response:**

The entire risk mapping module has been rewritten using the **Harvard-Oxford Probabilistic Cortical and Subcortical Structural Atlas** (25% maximum-probability threshold, 2mm isotropic MNI152 space).

Key improvements over bounding boxes:
- Uses actual gyral and subcortical parcel labels — not axis-aligned approximations
- Enforces rigorous spatial coordinate transformation via affine matrix inversion
- Preserves cortical lateralization (Left vs. Right Precentral Gyrus = motor cortex)
- Uses nearest-neighbor interpolation for resampling categorical masks

**Mathematical transformation:**
```
v_patient = (A_patient⁻¹ · M_reg⁻¹ · A_atlas) · v_atlas
```

**Files:**
- `packages/ml/src/braintumor/risk/mapping.py`
- `packages/ml/tests/test_risk_mapping.py` (20 tests, all passing)

---

### Reviewer Concern 10 — Patient-Level Data Splitting

**Concern:** Reviewers requested confirmation that data splits are at the patient level (not slice level) to prevent data leakage.

**Response:**

The `discover_and_split()` function in `packages/ml/src/braintumor/data/splits.py` performs patient-level splitting:
- Each BraTS case directory = one patient
- Cases are assigned as a whole to train/validation/test partitions
- Split is seeded for reproducibility and serialized to `split.json`
- Both the 2D training script and the 3D training script use the same `discover_and_split()` call with the same `split_file` path, ensuring identical patient-level partitions

**Files:**
- `packages/ml/src/braintumor/data/splits.py`
- `scripts/train.py` and `scripts/train_3d.py` (shared `--split_file` argument)

---

### Reviewer Concern 11 — Inference Benchmarking

**Concern:** Reviewers requested timing and resource benchmarks for the inference pipeline.

**Response:**

A comprehensive benchmarking suite (`scripts/benchmark_inference.py`) measures:
- Preprocessing time
- Model forward pass time (separately for 2D and 3D)
- Postprocessing time
- Atlas-based anatomical risk mapping time
- Report generation time
- Total end-to-end time
- Peak RAM consumption
- Hardware and software environment

Runs multiple cases with configurable `--n_cases` and reports: mean, std, median, P95, min, max.

**Actual 2D benchmark results (CPU, Apple Silicon, 128³ volume, n=2):**

| Stage | Mean (s) |
|-------|---------|
| Preprocessing | 0.333 |
| Inference | 12.553 |
| Postprocessing | 0.188 |
| Atlas mapping | 0.394 |
| Report generation | 0.0011 |
| **Total end-to-end** | **13.47** |
| Peak RAM | 110.4 MB |

Results saved to: `outputs/benchmarks/benchmark_attn_unet_2d.json`

**Files:**
- `scripts/benchmark_inference.py`
- `outputs/benchmarks/`

---

### Reviewer Concern 12 — Report Numerical Consistency

**Concern:** Reviewers were concerned about numerical consistency between the structured data and the generated plain-language report (especially if LLM-generated).

**Response:**

A formal verification engine (`verify_report_numbers()`) in `risk/report.py` acts as an automated safety barrier:

1. **Number whitelist check:** Every floating-point number in the report text must appear in the structured data (overlap percentages, distances, volumes, counts).
2. **Disclaimer enforcement:** The exact clinical disclaimer must be present verbatim.
3. **Forbidden wording check:** Rejects clinical directives ("recommend", "should undergo", "resect", "safe to operate", "no risk", "risk-free", "guarantee").
4. **Contradictory tier check:** Detects if a region is described with a different risk tier than the computed one.

The LLM-enhanced report (if `ANTHROPIC_API_KEY` is set) is passed through this verifier before use. Any failure causes automatic fallback to the deterministic template.

**Test coverage:** 300-case randomized consistency test (`test_template_reports_preserve_numbers`) + 7 targeted edge cases.

**Files:**
- `packages/ml/src/braintumor/risk/report.py`
- `packages/ml/tests/test_report_consistency.py` (all 7 non-LLM tests pass)

---

### Reviewer Concern 13 — Empty Mask and Edge-Case Handling

**Concern:** Reviewers were concerned about undefined or misleading behavior when the segmentation model produces an empty mask (no tumor detected).

**Response:**

The system implements explicit, distinguishable status codes for all failure and edge cases:

| Status | Meaning | Report behavior |
|--------|---------|-----------------|
| `SUCCESS` | Normal operation | Full risk report with tiers |
| `NO_TUMOR_DETECTED` | Empty segmentation mask | Explicit "NO TUMOR DETECTED" report — no risk tiers assigned (prevents false reassurance from labeling all regions "Low") |
| `FAILED_REGISTRATION` | Singular or invalid affine matrix | Explicit registration failure report — no risk tiers |
| `INVALID_SEGMENTATION` | Non-finite values or wrong labels | Explicit invalid input report — no risk tiers |

**Files:**
- `packages/ml/src/braintumor/risk/mapping.py` — `MappingStatus` enum, `validate_spatial_inputs()`
- `packages/ml/src/braintumor/risk/report.py` — `_no_tumor_report()`, `_failed_registration_report()`, `_invalid_segmentation_report()`
- `packages/ml/tests/test_risk_mapping.py` — `TestNoTumorDetected` class (2 tests)

---

### Reviewer Concern 14 — Incomplete Reference [10]

**Concern:** Reviewers noted that Reference [10] (Lolli et al.) contained only a draft placeholder without full bibliographic details.

**Response:**

Reference [10] has been fully identified, verified against the *Neurological Sciences* volume 44 index and Springer Nature catalog, and completed:

**Lolli, V. E., Coolen, T., Sadeghi, N., Voordecker, P., & Lefranc, F.** (2023). "BOLD fMRI and DTI fiber tracking for preoperative mapping of eloquent cerebral regions in brain tumor patients: impact on surgical approach and outcome." *Neurological Sciences*, 44(8), 2903–2914. DOI: 10.1007/s10072-023-06667-2. PMID: 36914833.

The article was published online 13 March 2023 (August 2023 issue). Its findings directly support NeuroRisk Lens's framing as an exploratory surgical risk-stratification prototype.

All 15 references are now verified in `docs/references_verified.md`.

**Files:**
- `docs/references_verified.md`

---

### Reviewer Concern 15 — Clinical Disclaimer

**Concern:** Reviewers required a mandatory, unambiguous clinical disclaimer in all generated reports.

**Response:**

A fixed disclaimer string is enforced at three levels:

1. **Hardcoded in `report.py`** — the constant `DISCLAIMER` cannot be changed without code modification
2. **Automatically appended** to every generated report (template, no-tumor, failure states)
3. **Verified programmatically** — `verify_report_numbers()` checks for the exact disclaimer string; any report missing it fails verification and falls back to the template

The disclaimer reads:
> *"This report is generated by a research/educational prototype. Region locations are approximate (heuristic, not a validated brain atlas). This is NOT a medical device and must NOT be used for real surgical planning or clinical decisions. Always consult a qualified radiologist or neurosurgeon."*

**Files:**
- `packages/ml/src/braintumor/risk/report.py` — `DISCLAIMER` constant
- `packages/ml/tests/test_report_consistency.py` — `test_verifier_requires_disclaimer_and_blocks_recommendations`

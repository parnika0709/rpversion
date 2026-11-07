"""Reproducible Comprehensive Segmentation and Risk Assessment Evaluation Script.

Developed in response to Reviewer Comments 2, 3, 4, 8, and 12.
Calculates:
  - Dice coefficient
  - Intersection over Union (IoU / Jaccard)
  - Precision (Positive Predictive Value)
  - Recall / Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)
  - 95th Percentile Hausdorff Distance (HD95, in mm)
  - Class-wise ({NCR/NET: 1, Edema: 2, Enhancing: 4}) and composite regions ({WT, TC, ET})
  - Aggregate statistics (mean, std, median, IQR)
  - Atlas-based Harvard-Oxford anatomical risk mapping
  - Explicit handling for empty masks, missing labels, and registration checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from braintumor.eval.metrics import (
    BRATS_CLASSES,
    BRATS_REGIONS,
    aggregate_metrics,
    evaluate_case,
)
from braintumor.inference.postprocess import postprocess
from braintumor.models.attention_unet_2d import MVP_MODALITIES, AttentionUnet2DPredictor
from braintumor.models.unet_3d import Unet3DPredictor
from braintumor.risk.mapping import compute_risk_regions
from braintumor.risk.report import build_report_text
from braintumor.schemas.contracts import VolumeInput, VolumeModality


def evaluate_single_patient(
    predictor,
    t1ce_path: Path,
    t2_path: Path,
    flair_path: Path,
    seg_path: Path | None = None,
    output_dir: Path | None = None,
    case_id: str = "evaluated_case",
    reorient_ras: bool = False,
) -> dict[str, Any]:
    """Run full evaluation on a complete multimodal MRI volume."""
    print(f"\nEvaluating patient: {case_id}")
    print(f"  T1ce:  {t1ce_path}")
    print(f"  T2:    {t2_path}")
    print(f"  FLAIR: {flair_path}")
    if seg_path:
        print(f"  GT Seg:{seg_path}")

    # Load NIfTI volumes
    t1ce_img = nib.load(str(t1ce_path))
    t2_img = nib.load(str(t2_path))
    flair_img = nib.load(str(flair_path))

    orig_affine = np.asarray(t1ce_img.affine, dtype=np.float64)
    orig_spacing = tuple(float(s) for s in nib.affines.voxel_sizes(orig_affine)[:3])
    orig_orientation = nib.orientations.aff2axcodes(orig_affine)

    if reorient_ras:
        print(f"  [PHASE 2] Reorienting MRI volumes from {orig_orientation} to canonical RAS...")
        t1ce_img = nib.as_closest_canonical(t1ce_img)
        t2_img = nib.as_closest_canonical(t2_img)
        flair_img = nib.as_closest_canonical(flair_img)

    t1ce_data = np.asarray(t1ce_img.get_fdata(), dtype=np.float32)
    t2_data = np.asarray(t2_img.get_fdata(), dtype=np.float32)
    flair_data = np.asarray(flair_img.get_fdata(), dtype=np.float32)

    affine = np.asarray(t1ce_img.affine, dtype=np.float64)
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(affine)[:3])
    current_orientation = nib.orientations.aff2axcodes(affine)

    # PHASE 3 Validation Checks before inference
    if reorient_ras:
        print("  [PHASE 3] Validating spatial consistency:")
        assert t1ce_data.shape == t2_data.shape == flair_data.shape, (
            f"Modality shapes mismatch: T1ce {t1ce_data.shape}, T2 {t2_data.shape}, FLAIR {flair_data.shape}"
        )
        assert np.allclose(t1ce_img.affine, t2_img.affine) and np.allclose(t1ce_img.affine, flair_img.affine), (
            "Transformed affines between modalities do not match!"
        )
        assert current_orientation == ("R", "A", "S"), (
            f"Expected canonical RAS orientation, got {current_orientation}"
        )
        assert np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-3), (
            f"Expected isotropic (1.0, 1.0, 1.0) spacing, got {spacing}"
        )
        print(f"    - Modality shapes aligned: {t1ce_data.shape}")
        print(f"    - Modality affines matched (RAS diagonal: {affine.diagonal()[:3]})")
        print(f"    - Orientation confirmed: {current_orientation}")
        print(f"    - Voxel spacing confirmed: {spacing}")

    # Preprocessing: Percentile intensity scaling to [0, 1] + non-zero z-score normalization
    # Guaranteed consistent with MONAI's ScaleIntensityRangePercentilesd + NormalizeIntensityd(nonzero=True)
    modalities_data = [t1ce_data, t2_data, flair_data]
    processed_channels = []
    for arr in modalities_data:
        p1, p99 = np.percentile(arr, (1.0, 99.0))
        scaled = np.clip(arr, p1, p99)
        if p99 > p1:
            scaled = (scaled - p1) / (p99 - p1)
        else:
            scaled = np.zeros_like(scaled)
        mask = scaled > 0
        if mask.any():
            mean = scaled[mask].mean()
            std = scaled[mask].std() + 1e-8
            scaled[mask] = (scaled[mask] - mean) / std
            scaled[~mask] = 0.0
        # Phase 3 assertion: verify background voxels remain strictly 0.0
        assert (scaled[arr == 0] == 0.0).all(), "Background voxels non-zero after normalization!"
        processed_channels.append(scaled)

    if reorient_ras:
        print("    - Intensity normalization confirmed: background strictly 0.0 across all channels.")

    model_input_arr = np.stack(processed_channels, axis=0)

    volume = VolumeInput(
        modalities=[VolumeModality.T1CE, VolumeModality.T2, VolumeModality.FLAIR],
        array=model_input_arr,
        spacing=spacing,
        affine=affine,
    )

    # 1. Model Prediction
    print("  Running model inference...")
    pred_output = predictor.predict(volume)
    raw_mask = pred_output.mask
    processed_mask = postprocess(raw_mask)

    results: dict[str, Any] = {
        "case_id": case_id,
        "input_shape": list(t1ce_data.shape),
        "spacing_mm": list(spacing),
        "reorient_ras": reorient_ras,
        "orig_orientation": list(orig_orientation),
        "current_orientation": list(current_orientation),
        "orig_affine": orig_affine.tolist(),
        "affine": affine.tolist(),
        "inference_seconds": pred_output.meta.inference_seconds,
        "tumor_voxels_raw": int((raw_mask > 0).sum()),
        "tumor_voxels_postprocessed": int((processed_mask > 0).sum()),
    }

    # 2. Ground-Truth Segmentation Metrics (if ground truth available)
    if seg_path and Path(seg_path).is_file():
        gt_img = nib.load(str(seg_path))
        if reorient_ras:
            gt_img = nib.as_closest_canonical(gt_img)
        gt_data = np.asarray(gt_img.get_fdata(), dtype=np.uint8)

        if reorient_ras:
            assert gt_data.shape == t1ce_data.shape, "Segmentation shape does not match MRI volume shape!"
            assert np.allclose(gt_img.affine, t1ce_img.affine), "Segmentation affine does not match MRI affine!"
            assert np.isin(gt_data, [0, 1, 2, 4]).all(), "Invalid segmentation labels found after reorientation!"
            print(f"    - Ground-truth segmentation aligned: {gt_data.shape}, labels: {np.unique(gt_data).tolist()}")

        print("  Calculating full segmentation metric battery...")
        # Metrics on composite BraTS regions (WT, TC, ET)
        raw_region_metrics = evaluate_case(raw_mask, gt_data, spacing, regions=BRATS_REGIONS)
        post_region_metrics = evaluate_case(processed_mask, gt_data, spacing, regions=BRATS_REGIONS)

        # Metrics on individual classes (NCR/NET: 1, Edema: 2, Enhancing: 4)
        raw_class_metrics = evaluate_case(raw_mask, gt_data, spacing, regions=BRATS_CLASSES)
        post_class_metrics = evaluate_case(processed_mask, gt_data, spacing, regions=BRATS_CLASSES)

        def metrics_to_dict(rm_dict: dict) -> dict:
            return {
                k: {
                    "dice": m.dice,
                    "iou": m.iou,
                    "sensitivity": m.sensitivity,
                    "recall": m.recall,
                    "specificity": m.specificity,
                    "precision": m.precision,
                    "hausdorff95_mm": m.hausdorff95_mm,
                }
                for k, m in rm_dict.items()
            }

        results["raw_metrics"] = {
            "regions": metrics_to_dict(raw_region_metrics),
            "classes": metrics_to_dict(raw_class_metrics),
        }
        results["postprocessed_metrics"] = {
            "regions": metrics_to_dict(post_region_metrics),
            "classes": metrics_to_dict(post_class_metrics),
        }

        gt_tumor_voxels = int((gt_data > 0).sum())
        pred_tumor_voxels = int((processed_mask > 0).sum())
        fp_voxels = int(((processed_mask > 0) & (gt_data == 0)).sum())
        fn_voxels = int(((processed_mask == 0) & (gt_data > 0)).sum())
        tp_voxels = int(((processed_mask > 0) & (gt_data > 0)).sum())
        voxel_vol = float(np.prod(spacing))
        gt_vol_mm3 = float(gt_tumor_voxels * voxel_vol)
        pred_vol_mm3 = float(pred_tumor_voxels * voxel_vol)

        results["volume_analysis"] = {
            "gt_tumor_voxels": gt_tumor_voxels,
            "pred_tumor_voxels": pred_tumor_voxels,
            "tp_voxels": tp_voxels,
            "fp_voxels": fp_voxels,
            "fn_voxels": fn_voxels,
            "gt_volume_mm3": gt_vol_mm3,
            "pred_volume_mm3": pred_vol_mm3,
        }

        # Print metrics table
        print("\n  SEGMENTATION METRICS SUMMARY:")
        print(f"  {'Region/Class':<12} | {'Dice':<8} | {'IoU':<8} | {'Sens/Rec':<8} | {'Spec':<8} | {'Prec':<8} | {'HD95 (mm)':<8}")
        print("  " + "-" * 72)
        for r_name in ("WT", "TC", "ET"):
            m = post_region_metrics[r_name]
            hd_str = f"{m.hausdorff95_mm:.2f}" if not np.isinf(m.hausdorff95_mm) and not np.isnan(m.hausdorff95_mm) else "inf"
            print(f"  {r_name:<12} | {m.dice:<8.4f} | {m.iou:<8.4f} | {m.sensitivity:<8.4f} | {m.specificity:<8.4f} | {m.precision:<8.4f} | {hd_str:<8}")
        print("  " + "-" * 72)
        print(f"  GT Tumor Voxels:   {gt_tumor_voxels:,} ({gt_vol_mm3:,.1f} mm³)")
        print(f"  Pred Tumor Voxels: {pred_tumor_voxels:,} ({pred_vol_mm3:,.1f} mm³)")
        print(f"  True Positives:    {tp_voxels:,} voxels")
        print(f"  False Positives:   {fp_voxels:,} voxels")
        print(f"  False Negatives:   {fn_voxels:,} voxels")

    # 3. Atlas Risk Mapping
    print("  Running Harvard-Oxford MNI152 anatomical mapping...")
    risk_regions = compute_risk_regions(
        tumor_mask=processed_mask,
        affine=affine,
        spacing=spacing,
    )

    results["anatomical_risk_mapping"] = [
        {
            "region": r.region,
            "function": r.function,
            "risk_tier": r.risk_tier,
            "overlap_pct": r.overlap_pct,
            "region_overlap_pct": r.region_overlap_pct,
            "distance_mm": r.distance_mm,
            "overlap_volume_mm3": r.overlap_volume_mm3,
            "region_volume_mm3": r.region_volume_mm3,
            "mapping_status": r.mapping_status,
        }
        for r in risk_regions
    ]

    # 4. Report Text
    tumor_vol_mm3 = float((processed_mask > 0).sum() * np.prod(spacing))
    report_text = build_report_text(case_id, risk_regions, tumor_vol_mm3)
    results["report_text"] = report_text

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save predicted mask
        mask_nii = nib.Nifti1Image(processed_mask.astype(np.uint8), affine)
        nib.save(mask_nii, str(out_dir / f"{case_id}_pred_seg.nii.gz"))

        # Save structured results
        with open(out_dir / f"{case_id}_eval_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save report text
        with open(out_dir / f"{case_id}_report.txt", "w") as f:
            f.write(report_text)

        print(f"  Saved evaluation artifacts to: {out_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive BraTS Segmentation & Risk Evaluation")
    parser.add_argument("--model", type=str, default="attn_unet_2d", choices=["attn_unet_2d", "unet_3d"])
    parser.add_argument("--checkpoint", type=str, default="runs/exp/best.pt")
    parser.add_argument("--case_dir", type=str, default="/Users/pvarish/Downloads/BraTS2021_00495")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--roi_size", type=int, default=64, help="ROI size for 3D sliding window")
    parser.add_argument(
        "--reorient_ras",
        action="store_true",
        help="Reorient all modalities and ground truth to canonical RAS to match 3D training coordinate space.",
    )

    args = parser.parse_args()

    # Load predictor
    if args.model == "attn_unet_2d":
        predictor = AttentionUnet2DPredictor(
            weights_path=Path(args.checkpoint) if Path(args.checkpoint).is_file() else None,
            device=args.device,
        )
    else:
        predictor = Unet3DPredictor(
            weights_path=Path(args.checkpoint) if Path(args.checkpoint).is_file() else None,
            device=args.device,
            roi_size=(args.roi_size, args.roi_size, args.roi_size),
        )

    case_path = Path(args.case_dir)
    if not case_path.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_path}")

    # Discover modality files
    flair = next((f for f in case_path.iterdir() if "flair" in f.name.lower() and f.name.endswith((".nii", ".nii.gz"))), None)
    t1ce = next((f for f in case_path.iterdir() if "t1ce" in f.name.lower() and f.name.endswith((".nii", ".nii.gz"))), None)
    t2 = next((f for f in case_path.iterdir() if "t2" in f.name.lower() and "t1" not in f.name.lower() and f.name.endswith((".nii", ".nii.gz"))), None)
    seg = next((f for f in case_path.iterdir() if "seg" in f.name.lower() or "groundtruth" in f.name.lower()), None)

    if not flair or not t1ce or not t2:
        raise ValueError(f"Missing required modalities in {case_path}. Found: flair={flair}, t1ce={t1ce}, t2={t2}")

    evaluate_single_patient(
        predictor=predictor,
        t1ce_path=t1ce,
        t2_path=t2,
        flair_path=flair,
        seg_path=seg,
        output_dir=Path(args.output_dir),
        case_id=case_path.name,
        reorient_ras=args.reorient_ras,
    )


if __name__ == "__main__":
    main()

"""Threshold sensitivity (and prediction-vs-ground-truth tier agreement).

Usage (from the project root):

    python3 scripts/risk_sensitivity.py \
        --mask runs/results/case_25/mask.nii.gz \
        --t1ce "uploads/<case>/<name>_t1ce.nii.gz" \
        --gt   "/path/to/BraTS2021_00495_seg.nii.gz" \
        --out  sensitivity_case25.csv

--mask : predicted tumor mask (any non-zero voxel = tumor)
--t1ce : the case's T1ce scan (used for the brain bounding box and spacing)
--gt   : optional ground-truth segmentation; if given, the script also
         reports how often the risk tiers computed from the predicted mask
         agree with those computed from the ground truth.
--out  : optional CSV of the sensitivity grid.

All three files must share the same voxel grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from braintumor.risk.mapping import (
    RiskThresholds,
    _risk_tier,
    compute_region_metrics,
    risk_tier_sensitivity,
)


def _load_mask(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--t1ce", required=True)
    parser.add_argument("--gt")
    parser.add_argument("--out")
    args = parser.parse_args()

    t1ce_img = nib.load(args.t1ce)
    volume = np.asarray(t1ce_img.get_fdata(), dtype=np.float32)[None]  # (1,D,H,W)
    spacing = tuple(float(z) for z in t1ce_img.header.get_zooms()[:3])

    pred = _load_mask(args.mask)
    if pred.shape != volume.shape[1:]:
        raise SystemExit(f"Mask {pred.shape} and T1ce {volume.shape[1:]} differ in shape.")

    pred_metrics = compute_region_metrics(pred, volume, spacing)
    rows = risk_tier_sensitivity(pred_metrics)

    print(f"Predicted tumor voxels: {int(pred.sum()):,}")
    print()
    print(f"{'overlap>=%':>11} {'dist<=mm':>9} {'High':>5} {'Mod':>5} {'Low':>5}")
    for r in rows:
        print(
            f"{r['high_overlap_pct']:>11.1f} {r['moderate_distance_mm']:>9.1f} "
            f"{r['High']:>5} {r['Moderate']:>5} {r['Low']:>5}"
        )

    if args.out:
        names = [m["region"] for m in pred_metrics]
        with open(args.out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["high_overlap_pct", "moderate_distance_mm", "High", "Moderate", "Low", *names])
            for r in rows:
                writer.writerow(
                    [r["high_overlap_pct"], r["moderate_distance_mm"], r["High"], r["Moderate"], r["Low"],
                     *[r["tiers"][n] for n in names]]
                )
        print(f"\nWrote {Path(args.out).resolve()}")

    if args.gt:
        gt = _load_mask(args.gt)
        if gt.shape != pred.shape:
            raise SystemExit(f"Ground truth {gt.shape} and mask {pred.shape} differ in shape.")
        gt_metrics = compute_region_metrics(gt, volume, spacing)
        default = RiskThresholds()
        agree = 0
        print("\nTier from prediction vs. ground truth (default thresholds):")
        for p, g in zip(pred_metrics, gt_metrics):
            tp = _risk_tier(p["overlap_pct"], p["distance_mm"], default)
            tg = _risk_tier(g["overlap_pct"], g["distance_mm"], default)
            agree += tp == tg
            print(f"  {p['region']:<20} pred={tp:<9} gt={tg:<9} {'OK' if tp == tg else 'DIFF'}")
        print(f"Agreement: {agree}/{len(pred_metrics)} regions ({100 * agree / len(pred_metrics):.0f}%)")


if __name__ == "__main__":
    main()
"""Segmentation evaluation metrics.

Reported per BraTS region (WT, TC, ET):
  - Dice coefficient (F1 of overlap)
  - IoU (Jaccard)
  - Hausdorff distance (95th percentile, mm) — surface distance, in mm
  - Sensitivity (recall)
  - Precision

All metrics are computed in physical space using the volume's voxel spacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

# BraTS native regions (see schemas/contracts.py).
BRATS_REGIONS = {
    "WT": (1, 2, 4),     # whole tumor
    "TC": (1, 4),        # tumor core
    "ET": (4,),          # enhancing tumor
}


@dataclass
class RegionMetrics:
    dice: float
    iou: float
    hausdorff95_mm: float
    sensitivity: float
    precision: float


def _binary_mask(label: np.ndarray, region: Iterable[int]) -> np.ndarray:
    out = np.zeros_like(label, dtype=bool)
    for v in region:
        out |= label == v
    return out


def dice(pred: np.ndarray, target: np.ndarray, region: Iterable[int]) -> float:
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    intersection = np.logical_and(p, t).sum()
    denom = p.sum() + t.sum()
    if denom == 0:
        return float("nan")  # region absent in both — undefined
    return float(2.0 * intersection / denom)


def iou(pred: np.ndarray, target: np.ndarray, region: Iterable[int]) -> float:
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    if union == 0:
        return float("nan")
    return float(intersection / union)


def sensitivity(pred: np.ndarray, target: np.ndarray, region: Iterable[int]) -> float:
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    if t.sum() == 0:
        return float("nan")
    return float(np.logical_and(p, t).sum() / t.sum())


def precision(pred: np.ndarray, target: np.ndarray, region: Iterable[int]) -> float:
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    if p.sum() == 0:
        return float("nan")
    return float(np.logical_and(p, t).sum() / p.sum())


def hausdorff95_mm(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    region: Iterable[int],
) -> float:
    """95th-percentile Hausdorff distance in mm, with physical spacing.

    Returns ``inf`` if either mask is empty (a meaningful boundary does not
    exist) and ``nan`` if both are empty (region absent in both).
    """
    p = _binary_mask(pred, region)
    t = _binary_mask(target, region)
    if not p.any() and not t.any():
        return float("nan")
    if not p.any() or not t.any():
        return float("inf")

    # Boundary voxels = surface of each mask (boundary - eroded).
    p_boundary = p ^ ndimage.binary_erosion(p)
    t_boundary = t ^ ndimage.binary_erosion(t)

    p_coords = np.argwhere(p_boundary) * np.asarray(spacing)
    t_coords = np.argwhere(t_boundary) * np.asarray(spacing)
    if p_coords.size == 0 or t_coords.size == 0:
        return float("nan")

    # Compute pairwise distances in chunks to avoid huge allocations.
    # For 95th percentile we don't need all pairs — random sample cap.
    max_pairs = 5_000_000
    if len(p_coords) * len(t_coords) > max_pairs:
        rng = np.random.default_rng(0)
        p_idx = rng.choice(len(p_coords), size=max_pairs // len(t_coords), replace=False)
        p_coords = p_coords[p_idx]

    diff = p_coords[:, None, :] - t_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    forward = np.percentile(dist.min(axis=1), 95)
    backward = np.percentile(dist.min(axis=0), 95)
    return float(max(forward, backward))


def evaluate_case(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, RegionMetrics]:
    """Compute metrics for all BraTS regions on a single case."""
    out: dict[str, RegionMetrics] = {}
    for region_name, region_labels in BRATS_REGIONS.items():
        out[region_name] = RegionMetrics(
            dice=dice(pred, target, region_labels),
            iou=iou(pred, target, region_labels),
            hausdorff95_mm=hausdorff95_mm(pred, target, spacing, region_labels),
            sensitivity=sensitivity(pred, target, region_labels),
            precision=precision(pred, target, region_labels),
        )
    return out


def aggregate_metrics(per_case: list[dict[str, RegionMetrics]]) -> dict[str, dict[str, float]]:
    """Aggregate per-case metrics into mean/std per region, ignoring NaNs."""
    aggregated: dict[str, dict[str, float]] = {}
    for region_name in BRATS_REGIONS:
        region_vals: dict[str, list[float]] = {
            "dice": [],
            "iou": [],
            "hausdorff95_mm": [],
            "sensitivity": [],
            "precision": [],
        }
        for case in per_case:
            rm = case[region_name]
            for metric_name, value in [
                ("dice", rm.dice),
                ("iou", rm.iou),
                ("hausdorff95_mm", rm.hausdorff95_mm),
                ("sensitivity", rm.sensitivity),
                ("precision", rm.precision),
            ]:
                if not np.isnan(value) and not np.isinf(value):
                    region_vals[metric_name].append(value)

        aggregated[region_name] = {
            metric: {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if vals else float("nan"),
                "n": len(vals),
            }
            for metric, vals in region_vals.items()
        }
    return aggregated

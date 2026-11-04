"""Heuristic surgical risk mapping.

This is a DELIBERATELY SIMPLE stand-in for real atlas registration
(SynthSeg / FSL / ANTsPy are heavy, slow, and not installed here).

Instead of registering a real brain atlas, we approximate each functional
region as a fixed fractional bounding box relative to the patient's own
brain bounding box (the box containing all non-zero voxels across
modalities). This is coarse and NOT anatomically precise, but it is fast,
has zero extra dependencies, and is good enough to demonstrate the idea:
"how close is the tumor to a functionally important area?"

Fractions are rough, commonly-cited approximate locations of each region
within a standard axial brain volume, expressed as (start, end) along each
axis in [0, 1]:
  - d: depth / axial slices (0 = inferior/bottom, 1 = superior/top)
  - h: height (0 = posterior/back, 1 = anterior/front)   [approximation]
  - w: width (0 = one side, 1 = other side)

NOT clinically validated. Do not use for real surgical planning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

# name -> (d_range, h_range, w_range), each a (start, end) fraction in [0, 1]
FUNCTIONAL_REGIONS: dict[str, dict] = {
    "motor_cortex_left": {
        "box": ((0.55, 0.85), (0.35, 0.65), (0.0, 0.45)),
        "function": "controls voluntary movement of the right side of the body",
    },
    "motor_cortex_right": {
        "box": ((0.55, 0.85), (0.35, 0.65), (0.55, 1.0)),
        "function": "controls voluntary movement of the left side of the body",
    },
    "broca_area": {
        "box": ((0.35, 0.65), (0.55, 0.85), (0.0, 0.4)),
        "function": "speech production (language expression)",
    },
    "wernicke_area": {
        "box": ((0.35, 0.6), (0.25, 0.55), (0.05, 0.45)),
        "function": "language comprehension",
    },
    "visual_cortex": {
        "box": ((0.35, 0.65), (0.0, 0.25), (0.25, 0.75)),
        "function": "visual processing",
    },
    "brainstem": {
        "box": ((0.0, 0.3), (0.4, 0.6), (0.35, 0.65)),
        "function": "controls vital functions (breathing, heart rate, consciousness)",
    },
    "thalamus": {
        "box": ((0.3, 0.55), (0.4, 0.65), (0.35, 0.65)),
        "function": "sensory relay and consciousness regulation",
    },
}

RISK_TIERS = ("High", "Moderate", "Low")

# Separate status used when the model finds no tumor at all. This is NOT a
# risk tier: reporting every region as "Low" in that case would read as false
# reassurance, so regions are marked "not assessed" instead.
NO_TUMOR_TIER = "No tumor detected"


@dataclass(frozen=True)
class RiskThresholds:
    """Heuristic cut-offs for the risk tiers (illustrative, not validated).

    overlap_pct is the percentage of the TUMOR's voxels that fall inside the
    region box (not the percentage of the region covered).

      High     : overlap_pct >= high_overlap_pct
      Moderate : 0 < overlap_pct < high_overlap_pct,
                 or distance_mm <= moderate_distance_mm
      Low      : no overlap and distance_mm > moderate_distance_mm
    """

    high_overlap_pct: float = 5.0
    moderate_distance_mm: float = 10.0


def thresholds_from_env() -> RiskThresholds:
    """Allow overriding thresholds without editing code.

    Set RISK_HIGH_OVERLAP_PCT and/or RISK_MODERATE_DISTANCE_MM in the
    environment, for example:  RISK_HIGH_OVERLAP_PCT=1 uvicorn app.main:app
    """
    defaults = RiskThresholds()
    try:
        high = float(os.getenv("RISK_HIGH_OVERLAP_PCT", defaults.high_overlap_pct))
        dist = float(os.getenv("RISK_MODERATE_DISTANCE_MM", defaults.moderate_distance_mm))
    except ValueError:
        return defaults
    return RiskThresholds(high_overlap_pct=high, moderate_distance_mm=dist)


@dataclass
class RiskRegion:
    region: str
    function: str
    overlap_pct: float
    distance_mm: float | None
    risk_tier: str


def _brain_bbox(volume_array: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Bounding box (in voxel indices) of all non-zero signal across channels.

    volume_array: (C, D, H, W)
    Returns ((d0, d1), (h0, h1), (w0, w1)) as inclusive-exclusive ranges.
    """
    nonzero = np.any(volume_array > 0, axis=0)  # (D, H, W)
    if not nonzero.any():
        d, h, w = volume_array.shape[1:]
        return (0, d), (0, h), (0, w)
    coords = np.argwhere(nonzero)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return (int(mins[0]), int(maxs[0])), (int(mins[1]), int(maxs[1])), (int(mins[2]), int(maxs[2]))


def _fractional_box_to_voxels(
    frac_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    brain_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    out = []
    for (f0, f1), (b0, b1) in zip(frac_box, brain_bbox):
        span = b1 - b0
        out.append((b0 + int(f0 * span), b0 + int(f1 * span)))
    return tuple(out)  # type: ignore[return-value]


def _region_mask(
    shape: tuple[int, int, int],
    voxel_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    (d0, d1), (h0, h1), (w0, w1) = voxel_box
    mask[d0:d1, h0:h1, w0:w1] = True
    return mask


def _distance_mm(
    tumor_mask: np.ndarray,
    region_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Compute minimum voxel-level Euclidean distance in millimetres."""
    tumor = np.asarray(tumor_mask, dtype=bool)
    region = np.asarray(region_mask, dtype=bool)

    if tumor.shape != region.shape:
        raise ValueError(
            f"Tumor and region masks must have the same shape: "
            f"{tumor.shape} != {region.shape}"
        )

    if not tumor.any() or not region.any():
        return float("inf")

    if len(spacing) != 3 or any(
        not np.isfinite(v) or v <= 0 for v in spacing
    ):
        raise ValueError("Spacing must contain three positive finite values.")

    distance_map = distance_transform_edt(
        ~region,
        sampling=tuple(float(v) for v in spacing),
    )
    return float(distance_map[tumor].min())


def _risk_tier(
    overlap_pct: float,
    distance_mm: float,
    thresholds: RiskThresholds | None = None,
) -> str:
    """Graded heuristic tier (see RiskThresholds)."""
    t = thresholds or thresholds_from_env()
    if overlap_pct >= t.high_overlap_pct:
        return "High"
    if overlap_pct > 0:
        return "Moderate"
    if distance_mm <= t.moderate_distance_mm:
        return "Moderate"
    return "Low"


def compute_region_metrics(
    tumor_mask: np.ndarray,
    volume_array: np.ndarray,
    spacing: tuple[float, float, float],
) -> list[dict]:
    """Overlap % and distance (mm) per region, before any tiering.

    Returned separately so the tier thresholds can be varied afterwards
    (sensitivity analysis) without recomputing distance transforms.
    """
    tumor_binary = tumor_mask > 0
    brain_bbox = _brain_bbox(volume_array)
    shape = tumor_mask.shape
    tumor_voxels = int(tumor_binary.sum())

    metrics: list[dict] = []
    for name, cfg in FUNCTIONAL_REGIONS.items():
        voxel_box = _fractional_box_to_voxels(cfg["box"], brain_bbox)
        region_mask = _region_mask(shape, voxel_box)

        overlap_voxels = int(np.logical_and(tumor_binary, region_mask).sum())
        overlap_pct = (100.0 * overlap_voxels / tumor_voxels) if tumor_voxels else 0.0

        if tumor_voxels == 0:
            distance_mm = float("inf")
        elif overlap_voxels > 0:
            distance_mm = 0.0
        else:
            distance_mm = _distance_mm(tumor_binary, region_mask, spacing)

        metrics.append(
            {
                "name": name,
                "region": name.replace("_", " "),
                "function": cfg["function"],
                "overlap_pct": overlap_pct,
                "distance_mm": distance_mm,
            }
        )
    return metrics


def compute_risk_regions(
    tumor_mask: np.ndarray,
    volume_array: np.ndarray,
    spacing: tuple[float, float, float],
    thresholds: RiskThresholds | None = None,
) -> list[RiskRegion]:
    """Compute overlap/distance/risk-tier for every functional region.

    Parameters
    ----------
    tumor_mask : (D, H, W) bool or int array (non-zero = tumor)
    volume_array : (C, D, H, W) — used to find the brain's own bounding box
    spacing : voxel spacing in mm
    thresholds : tier cut-offs (defaults to RiskThresholds / environment)

    Returns
    -------
    list[RiskRegion], sorted High -> Moderate -> Low, then by distance.
    If the mask contains no tumor, every region gets the status
    NO_TUMOR_TIER ("No tumor detected") instead of a risk tier.
    """
    thresholds = thresholds or thresholds_from_env()
    metrics = compute_region_metrics(tumor_mask, volume_array, spacing)
    no_tumor = not bool((tumor_mask > 0).any())

    results: list[RiskRegion] = []
    for m in metrics:
        distance_mm = m["distance_mm"]
        tier = (
            NO_TUMOR_TIER
            if no_tumor
            else _risk_tier(m["overlap_pct"], distance_mm, thresholds)
        )
        results.append(
            RiskRegion(
                region=m["region"],
                function=m["function"],
                overlap_pct=round(m["overlap_pct"], 1),
                distance_mm=None if distance_mm == float("inf") else round(distance_mm, 1),
                risk_tier=tier,
            )
        )

    tier_order = {"High": 0, "Moderate": 1, "Low": 2, NO_TUMOR_TIER: 3}
    results.sort(key=lambda r: (tier_order[r.risk_tier], r.distance_mm if r.distance_mm is not None else 1e9))
    return results


def risk_tier_sensitivity(
    metrics: list[dict],
    overlap_grid: tuple[float, ...] = (1.0, 2.5, 5.0, 10.0),
    distance_grid: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> list[dict]:
    """Re-tier the same region metrics under a grid of thresholds.

    Returns one row per (high_overlap_pct, moderate_distance_mm) pair with
    tier counts and the per-region tiers, for a threshold sensitivity table.
    """
    rows: list[dict] = []
    for high in overlap_grid:
        for dist in distance_grid:
            t = RiskThresholds(high_overlap_pct=high, moderate_distance_mm=dist)
            tiers = {
                m["region"]: _risk_tier(m["overlap_pct"], m["distance_mm"], t)
                for m in metrics
            }
            counts = {tier: sum(1 for v in tiers.values() if v == tier) for tier in RISK_TIERS}
            rows.append(
                {
                    "high_overlap_pct": high,
                    "moderate_distance_mm": dist,
                    **counts,
                    "tiers": tiers,
                }
            )
    return rows
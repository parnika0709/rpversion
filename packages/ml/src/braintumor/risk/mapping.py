"""Atlas-based anatomical risk mapping for neurosurgical planning research.

This module replaces heuristic bounding-box approximations with a scientifically
grounded, atlas-based anatomical mapping pipeline using the Harvard-Oxford Cortical
and Subcortical Structural Atlas in MNI152 standard space.

Key Scientific & Anatomical Principles:
1. Standard Brain Atlas:
   Uses the Harvard-Oxford probabilistic anatomical atlas (thresholded at 25%
   maximum probability, 2mm isotropic resolution) defined in MNI152 space.
   Anatomical structures follow true gyral and subcortical morphology rather
   than axis-aligned bounding boxes.
2. Coordinate Frames & Registration:
   Does not assume that patient MRI and atlas share the same coordinate system.
   Transforms spatial coordinates via patient affine (A_patient), atlas affine (A_atlas),
   and registration transform (M_reg):
       v_patient = (A_patient^-1 * M_reg^-1 * A_atlas) * v_atlas
   The tumor segmentation mask is resampled into the atlas reference frame using
   nearest-neighbor interpolation to preserve discrete boundary labels.
3. Rigorous Overlap & Coverage Metrics:
   - Tumor coverage (%): Proportion of total tumor volume overlapping the region.
   - Region coverage (%): Proportion of total anatomical region volume invaded by tumor.
   - Overlap volume (mm³), Region volume (mm³), and Tumor volume (mm³).
4. Physically Meaningful Distance Mapping:
   Calculates minimum Euclidean distance in true millimeters using anisotropic/isotropic
   voxel spacing via Euclidean Distance Transform (EDT).
5. Explicit Tiering & Empty Mask Handling:
   Configurable risk tiers ('High', 'Moderate', 'Low'). If no tumor is present,
   regions are explicitly marked 'No tumor detected' rather than 'Low' to prevent
   false reassurance.

DISCLAIMER:
This is an academic research prototype, NOT a clinically validated medical device.
Do not use for real surgical planning or patient care decisions.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import affine_transform, binary_erosion, distance_transform_edt

logger = logging.getLogger(__name__)

# Standard risk tier labels
RISK_TIERS = ("High", "Moderate", "Low")

# Status string when no tumor is detected in the volume
NO_TUMOR_TIER = "No tumor detected"


class MappingStatus(str, Enum):
    """Explicit mapping execution status."""

    SUCCESS = "SUCCESS"
    NO_TUMOR_DETECTED = "NO_TUMOR_DETECTED"
    LOW_RISK_DETECTED = "LOW_RISK_DETECTED"
    INVALID_SEGMENTATION = "INVALID_SEGMENTATION"
    FAILED_MISSING_ATLAS = "FAILED_MISSING_ATLAS"
    FAILED_INVALID_AFFINE = "FAILED_INVALID_AFFINE"
    FAILED_INCOMPATIBLE_DIMENSIONS = "FAILED_INCOMPATIBLE_DIMENSIONS"
    FAILED_REGISTRATION = "FAILED_REGISTRATION"


class AtlasMappingError(RuntimeError):
    """Raised when atlas-based anatomical mapping fails."""


class AtlasNotFoundError(AtlasMappingError):
    """Raised when atlas image or label files cannot be located."""


class RegistrationError(AtlasMappingError):
    """Raised when spatial coordinate registration or matrix inversion fails."""


@dataclass(frozen=True)
class RiskThresholds:
    """Configurable cut-offs for risk tier classification.

    Attributes
    ----------
    high_overlap_pct : float
        Percentage of the TUMOR volume that must fall within the region for 'High' risk.
    moderate_distance_mm : float
        Distance in millimeters within which a non-overlapping tumor is classified as 'Moderate'.
    high_region_overlap_pct : float
        Percentage of the ANATOMICAL REGION invaded by tumor for 'High' risk.
    """

    high_overlap_pct: float = 5.0
    moderate_distance_mm: float = 10.0
    high_region_overlap_pct: float = 5.0


def thresholds_from_env() -> RiskThresholds:
    """Load risk classification thresholds from environment variables with safe defaults."""
    defaults = RiskThresholds()
    try:
        high_tumor = float(os.getenv("RISK_HIGH_OVERLAP_PCT", defaults.high_overlap_pct))
        mod_dist = float(os.getenv("RISK_MODERATE_DISTANCE_MM", defaults.moderate_distance_mm))
        high_region = float(os.getenv("RISK_HIGH_REGION_OVERLAP_PCT", defaults.high_region_overlap_pct))
    except ValueError as err:
        logger.warning("Invalid environment variable for RiskThresholds: %s; using defaults", err)
        return defaults
    return RiskThresholds(
        high_overlap_pct=high_tumor,
        moderate_distance_mm=mod_dist,
        high_region_overlap_pct=high_region,
    )


@dataclass
class RiskRegion:
    """Results of anatomical risk mapping for a single anatomical or functional region."""

    region: str
    function: str
    overlap_pct: float  # Percentage of tumor inside this region
    distance_mm: float | None
    risk_tier: str
    region_overlap_pct: float = 0.0  # Percentage of region volume affected by tumor
    overlap_volume_mm3: float = 0.0
    region_volume_mm3: float = 0.0
    tumor_volume_mm3: float = 0.0
    atlas_name: str = "Harvard-Oxford"
    atlas_label_ids: tuple[int, ...] = ()
    mapping_status: str = MappingStatus.SUCCESS.value


@dataclass(frozen=True)
class AtlasRegionDefinition:
    """Definition linking a functional/eloquent brain area to Harvard-Oxford atlas labels."""

    name: str
    display_name: str
    function: str
    atlas_source: str  # 'cortical_lateralized' or 'subcortical'
    label_indices: tuple[int, ...]  # 1-based voxel values in maxprob atlas NIfTI


# Canonical functional regions mapped to Harvard-Oxford cortical lateralized & subcortical atlases.
# Preserves exact region names and function descriptions for backward compatibility with reporting and UI.
CANONICAL_FUNCTIONAL_REGIONS: dict[str, AtlasRegionDefinition] = {
    "motor_cortex_left": AtlasRegionDefinition(
        name="motor_cortex_left",
        display_name="motor cortex left",
        function="controls voluntary movement of the right side of the body",
        atlas_source="cortical_lateralized",
        # Label 13: Left Precentral Gyrus
        label_indices=(13,),
    ),
    "motor_cortex_right": AtlasRegionDefinition(
        name="motor_cortex_right",
        display_name="motor cortex right",
        function="controls voluntary movement of the left side of the body",
        atlas_source="cortical_lateralized",
        # Label 14: Right Precentral Gyrus
        label_indices=(14,),
    ),
    "broca_area": AtlasRegionDefinition(
        name="broca_area",
        display_name="broca area",
        function="speech production (language expression)",
        atlas_source="cortical_lateralized",
        # Labels 9, 11: Left Inferior Frontal Gyrus, pars triangularis & pars opercularis
        label_indices=(9, 11),
    ),
    "wernicke_area": AtlasRegionDefinition(
        name="wernicke_area",
        display_name="wernicke area",
        function="language comprehension",
        atlas_source="cortical_lateralized",
        # Labels 19 (Left STG posterior), 41 (Left Angular Gyrus), 91 (Left Planum Temporale)
        label_indices=(19, 41, 91),
    ),
    "visual_cortex": AtlasRegionDefinition(
        name="visual_cortex",
        display_name="visual cortex",
        function="visual processing",
        atlas_source="cortical_lateralized",
        # Labels 47, 48 (Intracalcarine), 93, 94 (Supracalcarine), 95, 96 (Occipital Pole)
        label_indices=(47, 48, 93, 94, 95, 96),
    ),
    "brainstem": AtlasRegionDefinition(
        name="brainstem",
        display_name="brainstem",
        function="controls vital functions (breathing, heart rate, consciousness)",
        atlas_source="subcortical",
        # Label 8: Brain-Stem
        label_indices=(8,),
    ),
    "thalamus": AtlasRegionDefinition(
        name="thalamus",
        display_name="thalamus",
        function="sensory relay and consciousness regulation",
        atlas_source="subcortical",
        # Labels 4, 15: Left Thalamus and Right Thalamus
        label_indices=(4, 15),
    ),
}

# Extended anatomical regions available for expanded surgical risk evaluation
EXTENDED_FUNCTIONAL_REGIONS: dict[str, AtlasRegionDefinition] = {
    **CANONICAL_FUNCTIONAL_REGIONS,
    "sensory_cortex_left": AtlasRegionDefinition(
        name="sensory_cortex_left",
        display_name="sensory cortex left",
        function="primary somatosensory processing (touch/proprioception) of right body",
        atlas_source="cortical_lateralized",
        # Label 33: Left Postcentral Gyrus
        label_indices=(33,),
    ),
    "sensory_cortex_right": AtlasRegionDefinition(
        name="sensory_cortex_right",
        display_name="sensory cortex right",
        function="primary somatosensory processing (touch/proprioception) of left body",
        atlas_source="cortical_lateralized",
        # Label 34: Right Postcentral Gyrus
        label_indices=(34,),
    ),
    "supplementary_motor_area": AtlasRegionDefinition(
        name="supplementary_motor_area",
        display_name="supplementary motor area",
        function="planning and coordination of complex movement sequences",
        atlas_source="cortical_lateralized",
        # Labels 51, 52: Left & Right Juxtapositional Lobule Cortex
        label_indices=(51, 52),
    ),
    "hippocampus_left": AtlasRegionDefinition(
        name="hippocampus_left",
        display_name="hippocampus left",
        function="episodic memory formation and spatial navigation",
        atlas_source="subcortical",
        # Label 9: Left Hippocampus
        label_indices=(9,),
    ),
    "hippocampus_right": AtlasRegionDefinition(
        name="hippocampus_right",
        display_name="hippocampus right",
        function="episodic memory formation and spatial navigation",
        atlas_source="subcortical",
        # Label 19: Right Hippocampus
        label_indices=(19,),
    ),
}

# Backward compatibility alias
FUNCTIONAL_REGIONS = CANONICAL_FUNCTIONAL_REGIONS


class HarvardOxfordAtlas:
    """Manages loading, caching, and spatial alignment of Harvard-Oxford atlas data."""

    def __init__(self, atlas_dir: Path | str | None = None) -> None:
        self.atlas_dir = self._resolve_atlas_dir(atlas_dir)
        self._cortl_img: Any = None
        self._sub_img: Any = None
        self._cortl_data: np.ndarray | None = None
        self._sub_data: np.ndarray | None = None
        self._labels_cache: dict[str, dict[int, str]] = {}
        self._region_mask_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _resolve_atlas_dir(explicit_dir: Path | str | None) -> Path:
        """Find the atlas directory from explicit path, environment, package, or nilearn."""
        if explicit_dir is not None:
            p = Path(explicit_dir)
            has_files = (
                (p / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz").exists()
                or (p / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz").exists()
                or (p / "HarvardOxford" / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz").exists()
            )
            if not p.exists() or not has_files:
                raise AtlasNotFoundError(f"Atlas files not found in specified directory: {p}")
            return p

        candidates: list[Path] = []
        env_dir = os.getenv("NEURORISK_ATLAS_DIR")
        if env_dir:
            candidates.append(Path(env_dir))

        # Project packaged path
        package_dir = Path(__file__).parent / "atlases" / "harvard_oxford"
        candidates.append(package_dir)

        # Nilearn standard data directory fallback
        user_home = Path.home()
        candidates.append(user_home / "nilearn_data" / "fsl" / "data" / "atlases" / "HarvardOxford")
        candidates.append(user_home / "nilearn_data" / "fsl" / "data" / "atlases")

        for c in candidates:
            if c.exists() and (
                (c / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz").exists()
                or (c / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz").exists()
            ):
                logger.debug("Resolved Harvard-Oxford atlas directory: %s", c)
                return c

        raise AtlasNotFoundError(
            f"Harvard-Oxford atlas files not found. Looked in: {[str(p) for p in candidates]}. "
            f"Please verify atlas installation or set NEURORISK_ATLAS_DIR."
        )

    def _load_images(self) -> None:
        """Load NIfTI atlas volumes into memory."""
        if self._cortl_data is not None and self._sub_data is not None:
            return

        import nibabel as nib

        cortl_path = self.atlas_dir / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz"
        sub_path = self.atlas_dir / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"

        # Check subdirectories if nested
        if not cortl_path.exists():
            nested_cortl = self.atlas_dir / "HarvardOxford" / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz"
            if nested_cortl.exists():
                cortl_path = nested_cortl

        if not sub_path.exists():
            nested_sub = self.atlas_dir / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"
            if nested_sub.exists():
                sub_path = nested_sub

        if not cortl_path.exists() or not sub_path.exists():
            raise AtlasNotFoundError(
                f"Missing required atlas files in {self.atlas_dir}: "
                f"cortl_exists={cortl_path.exists()}, sub_exists={sub_path.exists()}"
            )

        self._cortl_img = nib.load(str(cortl_path))
        self._sub_img = nib.load(str(sub_path))
        self._cortl_data = np.asarray(self._cortl_img.dataobj, dtype=np.int32)
        self._sub_data = np.asarray(self._sub_img.dataobj, dtype=np.int32)

        # Basic dimension sanity check
        if self._cortl_data.shape != self._sub_data.shape:
            raise AtlasMappingError(
                f"Cortical {self._cortl_data.shape} and Subcortical {self._sub_data.shape} "
                f"atlas volumes have mismatched shapes."
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        self._load_images()
        assert self._cortl_data is not None
        return tuple(int(x) for x in self._cortl_data.shape)  # type: ignore[return-value]

    @property
    def affine(self) -> np.ndarray:
        self._load_images()
        return np.asarray(self._cortl_img.affine, dtype=np.float64)

    @property
    def spacing(self) -> tuple[float, float, float]:
        self._load_images()
        zooms = self._cortl_img.header.get_zooms()[:3]
        return tuple(float(s) for s in zooms)  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        """Physical volume of a single atlas voxel in mm³."""
        return abs(float(np.linalg.det(self.affine[:3, :3])))

    def get_region_mask(self, region_def: AtlasRegionDefinition) -> np.ndarray:
        """Return a cached boolean 3D mask for a specified anatomical region in atlas space."""
        if region_def.name in self._region_mask_cache:
            return self._region_mask_cache[region_def.name]

        self._load_images()
        assert self._cortl_data is not None and self._sub_data is not None

        if region_def.atlas_source == "cortical_lateralized":
            data = self._cortl_data
        elif region_def.atlas_source == "subcortical":
            data = self._sub_data
        else:
            raise ValueError(f"Unknown atlas source '{region_def.atlas_source}'")

        mask = np.isin(data, region_def.label_indices)
        self._region_mask_cache[region_def.name] = mask
        return mask


# Global singleton instance for efficient caching across requests
_GLOBAL_ATLAS: HarvardOxfordAtlas | None = None


def get_default_atlas() -> HarvardOxfordAtlas:
    """Return the cached default Harvard-Oxford atlas instance."""
    global _GLOBAL_ATLAS
    if _GLOBAL_ATLAS is None:
        _GLOBAL_ATLAS = HarvardOxfordAtlas()
    return _GLOBAL_ATLAS


def validate_spatial_inputs(
    tumor_mask: np.ndarray,
    affine: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Validate spatial inputs, dimensions, affine matrix invertibility, and voxel spacing.

    Returns
    -------
    tuple of (validated_mask_3d, validated_affine_4x4, validated_spacing_3tuple)
    """
    if not isinstance(tumor_mask, np.ndarray):
        tumor_mask = np.asarray(tumor_mask)

    if tumor_mask.ndim != 3:
        raise ValueError(
            f"Tumor mask must be a 3D volume (D, H, W). Received shape: {tumor_mask.shape} (ndim={tumor_mask.ndim})"
        )

    # Validate or construct spacing
    if spacing is not None:
        if len(spacing) != 3 or any(not np.isfinite(s) or s <= 0 for s in spacing):
            raise ValueError(f"Spacing must contain three positive finite numbers. Received: {spacing}")
        val_spacing = tuple(float(s) for s in spacing)
    elif affine is not None:
        val_spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    else:
        val_spacing = (1.0, 1.0, 1.0)

    # Validate or construct affine
    if affine is not None:
        val_affine = np.asarray(affine, dtype=np.float64)
        if val_affine.shape != (4, 4):
            raise ValueError(f"Affine matrix must have shape (4, 4). Received shape: {val_affine.shape}")
        if not np.all(np.isfinite(val_affine)):
            raise ValueError("Affine matrix contains non-finite (NaN or Inf) values.")
        det = float(np.linalg.det(val_affine[:3, :3]))
        if abs(det) < 1e-8:
            raise RegistrationError(
                f"Singular or near-singular affine matrix with determinant {det:.2e}. Matrix cannot be inverted."
            )
    else:
        # Construct standard centered RAS affine from voxel spacing and mask dimensions
        logger.debug("No affine provided; constructing standard centered RAS affine from spacing and shape.")
        val_affine = np.eye(4, dtype=np.float64)
        for i in range(3):
            val_affine[i, i] = val_spacing[i]
            val_affine[i, 3] = -0.5 * tumor_mask.shape[i] * val_spacing[i]

    return tumor_mask, val_affine, val_spacing


def resample_mask_to_atlas(
    tumor_mask: np.ndarray,
    patient_affine: np.ndarray,
    atlas_affine: np.ndarray,
    atlas_shape: tuple[int, int, int],
    registration_affine: np.ndarray | None = None,
) -> np.ndarray:
    """Resample patient tumor mask into atlas coordinate grid using nearest-neighbor interpolation.

    Mathematical Transformation:
    For an atlas voxel coordinate v_atlas = (i_a, j_a, k_a, 1)^T:
      x_atlas = A_atlas * v_atlas
      x_patient = M_reg^-1 * x_atlas
      v_patient = A_patient^-1 * x_patient = (A_patient^-1 * M_reg^-1 * A_atlas) * v_atlas

    Nearest-neighbor interpolation is mathematically required for binary/categorical
    segmentation masks to avoid creating artificial intermediate fractional labels.
    """
    if not tumor_mask.any():
        return np.zeros(atlas_shape, dtype=bool)

    if registration_affine is None:
        m_reg = np.eye(4, dtype=np.float64)
    else:
        m_reg = np.asarray(registration_affine, dtype=np.float64)
        if m_reg.shape != (4, 4) or abs(np.linalg.det(m_reg[:3, :3])) < 1e-8:
            raise RegistrationError("Invalid or singular registration matrix provided.")

    try:
        inv_patient = np.linalg.inv(patient_affine)
        inv_reg = np.linalg.inv(m_reg)
        m_pull = inv_patient @ inv_reg @ atlas_affine
    except np.linalg.LinAlgError as err:
        raise RegistrationError(f"Failed to invert spatial transformation matrices: {err}") from err

    # scipy.ndimage.affine_transform maps output coordinate y -> input coordinate matrix @ y + offset
    matrix = m_pull[:3, :3]
    offset = m_pull[:3, 3]

    resampled = affine_transform(
        tumor_mask.astype(np.uint8),
        matrix=matrix,
        offset=offset,
        output_shape=atlas_shape,
        order=0,  # Nearest-neighbor interpolation
        mode="constant",
        cval=0,
    )
    return resampled > 0


def surface_to_surface_distance_mm(
    tumor_atlas_mask: np.ndarray,
    region_atlas_mask: np.ndarray,
    atlas_spacing: tuple[float, float, float],
    method: str = "edt",
) -> float:
    """Compute the physically meaningful surface-to-surface distance in true millimeters.

    Parameters
    ----------
    tumor_atlas_mask : np.ndarray
        Boolean 3D mask of the tumor in atlas space.
    region_atlas_mask : np.ndarray
        Boolean 3D mask of the anatomical target region in atlas space.
    atlas_spacing : tuple of float
        Voxel dimensions (dx, dy, dz) in millimeters.
    method : str
        Algorithm used to compute distance:
        - 'edt' (default): Exact Euclidean Distance Transform computed over the
          complement of the region mask sampled with physical voxel spacing.
          Fast O(N) evaluation yielding true minimum Euclidean distance to region surface.
        - 'boundary_coords': Extracts boundary surface voxels via morphological
          erosion (mask ^ binary_erosion(mask)), projects coordinates into physical mm,
          and calculates exact minimum pairwise Euclidean distance.

    Returns
    -------
    float
        Distance in millimeters (0.0 if overlapping, inf if either mask is empty).
    """
    if not tumor_atlas_mask.any() or not region_atlas_mask.any():
        return float("inf")

    # If any voxels overlap, distance is exactly 0.0 mm
    if np.logical_and(tumor_atlas_mask, region_atlas_mask).any():
        return 0.0

    if method == "boundary_coords":
        # Extract boundary surface voxels
        t_bound = tumor_atlas_mask ^ binary_erosion(tumor_atlas_mask)
        r_bound = region_atlas_mask ^ binary_erosion(region_atlas_mask)

        # Fallback to full masks if degenerate 1-voxel mask has empty eroded boundary
        t_coords = np.argwhere(t_bound if t_bound.any() else tumor_atlas_mask)
        r_coords = np.argwhere(r_bound if r_bound.any() else region_atlas_mask)

        t_mm = t_coords * np.asarray(atlas_spacing)
        r_mm = r_coords * np.asarray(atlas_spacing)

        # Subsample if surface has a very large number of voxels
        max_pts = 10_000
        if len(t_mm) > max_pts:
            idx = np.random.default_rng(42).choice(len(t_mm), size=max_pts, replace=False)
            t_mm = t_mm[idx]

        min_dist = float("inf")
        # Chunked pairwise computation to maintain memory efficiency
        chunk_size = 1000
        for start in range(0, len(t_mm), chunk_size):
            end = min(start + chunk_size, len(t_mm))
            diff = t_mm[start:end, None, :] - r_mm[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=-1))
            chunk_min = float(dists.min())
            if chunk_min < min_dist:
                min_dist = chunk_min

        return min_dist

    # Default: EDT method
    distance_map = distance_transform_edt(
        ~region_atlas_mask,
        sampling=atlas_spacing,
    )
    return float(distance_map[tumor_atlas_mask].min())


def _distance_mm(
    tumor_atlas_mask: np.ndarray,
    region_atlas_mask: np.ndarray,
    atlas_spacing: tuple[float, float, float],
) -> float:
    """Compute minimum Euclidean distance in true millimeters from tumor to anatomical region."""
    return surface_to_surface_distance_mm(
        tumor_atlas_mask=tumor_atlas_mask,
        region_atlas_mask=region_atlas_mask,
        atlas_spacing=atlas_spacing,
        method="edt",
    )


def _risk_tier(
    overlap_pct: float,
    distance_mm: float,
    thresholds: RiskThresholds | None = None,
    region_overlap_pct: float = 0.0,
) -> str:
    """Classify risk tier ('High', 'Moderate', 'Low') based on configurable thresholds.

    Rules:
    - High     : Significant tumor overlap (>= high_overlap_pct) OR substantial
                 region invasion (>= high_region_overlap_pct).
    - Moderate : Direct overlap (> 0%) below the high threshold OR close proximity
                 (<= moderate_distance_mm).
    - Low      : No anatomical overlap (0%) and distance > moderate_distance_mm.
    """
    t = thresholds or thresholds_from_env()
    if overlap_pct >= t.high_overlap_pct or region_overlap_pct >= t.high_region_overlap_pct:
        return "High"
    if overlap_pct > 0 or region_overlap_pct > 0:
        return "Moderate"
    if distance_mm <= t.moderate_distance_mm:
        return "Moderate"
    return "Low"


def compute_region_metrics(
    tumor_mask: np.ndarray,
    volume_array: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
    affine: np.ndarray | None = None,
    registration_affine: np.ndarray | None = None,
    thresholds: RiskThresholds | None = None,
    regions: dict[str, AtlasRegionDefinition] | None = None,
    atlas: HarvardOxfordAtlas | None = None,
) -> list[dict[str, Any]]:
    """Compute rigorous atlas-based overlap metrics and distance measurements per anatomical region.

    Parameters
    ----------
    tumor_mask : np.ndarray
        (D, H, W) binary or integer segmentation mask.
    volume_array : np.ndarray, optional
        Reference anatomical volume (C, D, H, W) or (D, H, W), preserved for compatibility.
    spacing : tuple of float, optional
        Voxel spacing (dx, dy, dz) in mm.
    affine : np.ndarray, optional
        4x4 affine matrix defining patient scanner-to-world millimeter coordinate mapping.
    registration_affine : np.ndarray, optional
        4x4 affine matrix registering patient world space to atlas MNI152 space.
    thresholds : RiskThresholds, optional
        Tier cut-offs.
    regions : dict, optional
        Subset or custom dictionary of AtlasRegionDefinition to evaluate.
        Defaults to CANONICAL_FUNCTIONAL_REGIONS.
    atlas : HarvardOxfordAtlas, optional
        Custom atlas instance. Defaults to get_default_atlas().

    Returns
    -------
    list of dict with keys:
        'name', 'region', 'function', 'overlap_pct', 'region_overlap_pct',
        'overlap_volume_mm3', 'region_volume_mm3', 'tumor_volume_mm3',
        'distance_mm', 'risk_tier', 'atlas_name', 'atlas_label_ids'
    """
    tumor_mask, patient_affine, _ = validate_spatial_inputs(tumor_mask, affine=affine, spacing=spacing)
    atlas_obj = atlas or get_default_atlas()
    atlas_shape = atlas_obj.shape
    atlas_affine = atlas_obj.affine
    atlas_spacing = atlas_obj.spacing
    voxel_vol = atlas_obj.voxel_volume_mm3

    target_regions = regions or CANONICAL_FUNCTIONAL_REGIONS
    no_tumor = not bool(tumor_mask.any())

    # Map tumor mask into atlas coordinate grid
    if no_tumor:
        tumor_in_atlas = np.zeros(atlas_shape, dtype=bool)
    else:
        tumor_in_atlas = resample_mask_to_atlas(
            tumor_mask=tumor_mask,
            patient_affine=patient_affine,
            atlas_affine=atlas_affine,
            atlas_shape=atlas_shape,
            registration_affine=registration_affine,
        )

    tumor_voxels_atlas = int(tumor_in_atlas.sum())
    tumor_volume_mm3 = float(tumor_voxels_atlas * voxel_vol)

    metrics: list[dict[str, Any]] = []

    for name, r_def in target_regions.items():
        region_mask = atlas_obj.get_region_mask(r_def)
        region_voxels = int(region_mask.sum())
        region_volume_mm3 = float(region_voxels * voxel_vol)

        if no_tumor or tumor_voxels_atlas == 0:
            overlap_voxels = 0
            overlap_pct = 0.0
            region_overlap_pct = 0.0
            overlap_volume_mm3 = 0.0
            distance_mm = float("inf")
        else:
            overlap_mask = np.logical_and(tumor_in_atlas, region_mask)
            overlap_voxels = int(overlap_mask.sum())
            overlap_volume_mm3 = float(overlap_voxels * voxel_vol)

            overlap_pct = (100.0 * overlap_voxels / tumor_voxels_atlas) if tumor_voxels_atlas else 0.0
            region_overlap_pct = (100.0 * overlap_voxels / region_voxels) if region_voxels else 0.0

            if overlap_voxels > 0:
                distance_mm = 0.0
            else:
                distance_mm = _distance_mm(tumor_in_atlas, region_mask, atlas_spacing)

        metrics.append(
            {
                "name": name,
                "region": r_def.display_name,
                "function": r_def.function,
                "overlap_pct": overlap_pct,
                "region_overlap_pct": region_overlap_pct,
                "overlap_volume_mm3": overlap_volume_mm3,
                "region_volume_mm3": region_volume_mm3,
                "tumor_volume_mm3": tumor_volume_mm3,
                "distance_mm": distance_mm,
                "atlas_name": "Harvard-Oxford",
                "atlas_label_ids": r_def.label_indices,
            }
        )

    return metrics


def compute_risk_regions(
    tumor_mask: np.ndarray,
    volume_array: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
    affine: np.ndarray | None = None,
    registration_affine: np.ndarray | None = None,
    thresholds: RiskThresholds | None = None,
    regions: dict[str, AtlasRegionDefinition] | None = None,
    atlas: HarvardOxfordAtlas | None = None,
) -> list[RiskRegion]:
    """Compute atlas-based surgical risk classification for brain functional regions.

    Parameters
    ----------
    tumor_mask : (D, H, W) bool or int array (non-zero = tumor)
    volume_array : (C, D, H, W) or (D, H, W), optional
    spacing : voxel spacing in mm
    affine : (4, 4) patient NIfTI affine matrix
    registration_affine : (4, 4) registration transform from patient space to MNI152 space
    thresholds : tier cut-offs (defaults to RiskThresholds from environment)
    regions : optional custom dictionary of AtlasRegionDefinition
    atlas : optional custom HarvardOxfordAtlas instance

    Returns
    -------
    list[RiskRegion]
        Sorted High -> Moderate -> Low, then by distance.
        If the mask contains no tumor, every region receives NO_TUMOR_TIER
        ("No tumor detected") instead of false reassurance.
    """
    thresholds = thresholds or thresholds_from_env()
    no_tumor = not bool(np.asarray(tumor_mask > 0).any())

    metrics = compute_region_metrics(
        tumor_mask=tumor_mask,
        volume_array=volume_array,
        spacing=spacing,
        affine=affine,
        registration_affine=registration_affine,
        thresholds=thresholds,
        regions=regions,
        atlas=atlas,
    )

    results: list[RiskRegion] = []
    for m in metrics:
        dist = m["distance_mm"]
        if no_tumor:
            tier = NO_TUMOR_TIER
            final_dist = None
        else:
            tier = _risk_tier(
                overlap_pct=m["overlap_pct"],
                distance_mm=dist,
                thresholds=thresholds,
                region_overlap_pct=m["region_overlap_pct"],
            )
            final_dist = None if dist == float("inf") else round(dist, 1)

        results.append(
            RiskRegion(
                region=m["region"],
                function=m["function"],
                overlap_pct=round(m["overlap_pct"], 1),
                distance_mm=final_dist,
                risk_tier=tier,
                region_overlap_pct=round(m["region_overlap_pct"], 2),
                overlap_volume_mm3=round(m["overlap_volume_mm3"], 1),
                region_volume_mm3=round(m["region_volume_mm3"], 1),
                tumor_volume_mm3=round(m["tumor_volume_mm3"], 1),
                atlas_name=m["atlas_name"],
                atlas_label_ids=m["atlas_label_ids"],
                mapping_status=MappingStatus.NO_TUMOR_DETECTED.value if no_tumor else MappingStatus.SUCCESS.value,
            )
        )

    tier_order = {"High": 0, "Moderate": 1, "Low": 2, NO_TUMOR_TIER: 3}
    results.sort(key=lambda r: (tier_order.get(r.risk_tier, 4), r.distance_mm if r.distance_mm is not None else 1e9))
    return results


def risk_tier_sensitivity(
    metrics: list[dict[str, Any]],
    overlap_grid: tuple[float, ...] = (1.0, 2.5, 5.0, 10.0),
    distance_grid: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> list[dict[str, Any]]:
    """Re-tier region metrics under a grid of thresholds for sensitivity analysis."""
    rows: list[dict[str, Any]] = []
    for high in overlap_grid:
        for dist in distance_grid:
            t = RiskThresholds(high_overlap_pct=high, moderate_distance_mm=dist)
            tiers = {
                m["region"]: _risk_tier(
                    overlap_pct=m["overlap_pct"],
                    distance_mm=m["distance_mm"],
                    thresholds=t,
                    region_overlap_pct=m.get("region_overlap_pct", 0.0),
                )
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

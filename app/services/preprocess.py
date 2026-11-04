"""Model-input preprocessing for the 2D Attention U-Net.

Mirrors ``braintumor.data.preprocessing.val_transforms`` (what the model was
trained with), implemented with numpy/nibabel/scipy so the predicted mask can
be mapped back into the uploaded scan's own space:

    RAS orientation -> 1 mm spacing -> 1-99 percentile scaling to [0, 1]
    -> z-score over non-zero voxels (per modality)

Without this, raw scanner intensities go straight into a network trained on
normalized data and the output is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
from nibabel import orientations as ornt_mod
from scipy import ndimage as ndi

_TARGET_SPACING = 1.0


@dataclass
class PreprocessInfo:
    """Everything needed to undo the spatial steps for a mask."""

    original_shape: tuple[int, int, int]
    to_ras: np.ndarray        # orientation transform: original -> RAS
    from_ras: np.ndarray      # orientation transform: RAS -> original
    ras_shape: tuple[int, int, int]    # shape right after reorientation
    model_shape: tuple[int, int, int]  # shape after resampling to 1 mm
    resampled: bool
    ras_affine: np.ndarray    # affine of the oriented (pre-resampling) volume


def _scale_percentiles(vol: np.ndarray) -> np.ndarray:
    """ScaleIntensityRangePercentiles(lower=1, upper=99, b=[0,1], clip=True)."""
    a_min, a_max = np.percentile(vol, [1.0, 99.0])
    if a_max - a_min == 0:
        scaled = vol - a_min
    else:
        scaled = (vol - a_min) / (a_max - a_min)
    return np.clip(scaled, 0.0, 1.0)


def _zscore_nonzero(vol: np.ndarray) -> np.ndarray:
    """NormalizeIntensity(nonzero=True): z-score over non-zero voxels only."""
    out = vol.astype(np.float32, copy=True)
    nz = out != 0
    if not nz.any():
        return out
    mean = out[nz].mean()
    std = out[nz].std()
    if std == 0:
        std = 1.0
    out[nz] = (out[nz] - mean) / std
    return out


def _fit_shape(arr: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Crop / zero-pad ``arr`` so its spatial shape equals ``shape``."""
    out = np.zeros(shape, dtype=arr.dtype)
    slices_in = tuple(slice(0, min(a, s)) for a, s in zip(arr.shape, shape))
    out[slices_in] = arr[slices_in]
    return out


def prepare_model_input(
    raw: np.ndarray,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, PreprocessInfo]:
    """Preprocess a raw (C, X, Y, Z) volume exactly like ``val_transforms``.

    Returns
    -------
    model_array : (C, X', Y', Z') float32, ready for the predictor
    model_affine : affine matching ``model_array``
    info : state needed by :func:`restore_mask`
    """
    if raw.ndim != 4:
        raise ValueError(f"Expected (C, X, Y, Z); got {raw.shape}")

    original_shape = tuple(int(s) for s in raw.shape[1:])

    orig_ornt = ornt_mod.io_orientation(affine)
    ras_ornt = ornt_mod.axcodes2ornt(("R", "A", "S"))
    to_ras = ornt_mod.ornt_transform(orig_ornt, ras_ornt)
    from_ras = ornt_mod.ornt_transform(ras_ornt, orig_ornt)

    oriented = [
        ornt_mod.apply_orientation(raw[c], to_ras) for c in range(raw.shape[0])
    ]
    ras_shape = tuple(int(s) for s in oriented[0].shape)
    ras_affine = affine @ ornt_mod.inv_ornt_aff(to_ras, original_shape)

    # Voxel spacing along each RAS axis (original axis i -> to_ras[i, 0]).
    ras_spacing = [1.0, 1.0, 1.0]
    for i in range(3):
        ras_spacing[int(to_ras[i, 0])] = float(spacing[i])

    resampled = any(abs(s - _TARGET_SPACING) > 1e-3 for s in ras_spacing)
    if resampled:
        zoom = tuple(s / _TARGET_SPACING for s in ras_spacing)
        oriented = [ndi.zoom(v, zoom, order=1) for v in oriented]
    model_shape = tuple(int(s) for s in oriented[0].shape)

    channels = [_zscore_nonzero(_scale_percentiles(v)) for v in oriented]
    model_array = np.stack(channels, axis=0).astype(np.float32)

    model_affine = ras_affine.copy()
    if resampled:
        for axis in range(3):
            model_affine[:3, axis] = (
                model_affine[:3, axis] / ras_spacing[axis] * _TARGET_SPACING
            )

    info = PreprocessInfo(
        original_shape=original_shape,
        to_ras=to_ras,
        from_ras=from_ras,
        ras_shape=ras_shape,
        model_shape=model_shape,
        resampled=resampled,
        ras_affine=ras_affine,
    )
    return model_array, model_affine, info


def restore_mask(mask: np.ndarray, info: PreprocessInfo) -> np.ndarray:
    """Map a mask from model space back onto the uploaded scan's grid."""
    mask = np.asarray(mask)

    if info.resampled:
        factors = tuple(
            t / c for t, c in zip(info.ras_shape, mask.shape)
        )
        mask = ndi.zoom(mask, factors, order=0)
        mask = _fit_shape(mask, info.ras_shape)

    restored = ornt_mod.apply_orientation(mask, info.from_ras)
    if tuple(restored.shape) != info.original_shape:
        restored = _fit_shape(restored, info.original_shape)
    return np.ascontiguousarray(restored)
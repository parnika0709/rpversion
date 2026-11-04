"""Service for running ML inference on MRI volumes."""

from __future__ import annotations

import os
import time
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Any
from scipy import ndimage as ndi
from types import SimpleNamespace

from app.config import settings
from app.db import Prediction, get_session
from app.services.preprocess import prepare_model_input, restore_mask
from braintumor.registry import registry
from braintumor.schemas.contracts import VolumeInput, VolumeModality
from braintumor.inference.pipeline import run_pipeline
from braintumor.features.extract import extract_features
from braintumor.risk import compute_risk_regions, build_report_text

# DEMO ONLY: set DEMO_MODE=1 to bypass the model and use a threshold mask.
# Tuned for the synthetic SYNTH_Patient_002 files, NOT for real scans.
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

# Project root (app/services/inference.py -> up 2 levels).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Register the only predictor we ship, once, at import time.
# (The registry raises if you register the same name twice, so guard it.)
if "attn_unet_2d" not in registry.list_names():
    from braintumor.models.attention_unet_2d import build_attn_unet_2d

    registry.register("attn_unet_2d", build_attn_unet_2d)


def _demo_mask(vol: np.ndarray) -> np.ndarray:
    """DEMO ONLY: fake 'prediction' by thresholding, in BraTS labels:
    1 = whole-tumor (edema), 2 = tumor core (necrotic), 4 = enhancing."""
    t1ce, _t2, flair = vol[0], vol[1], vol[2]
    hot = ndi.binary_opening((flair > 1100) | (t1ce > 1100), iterations=2)
    lab, n = ndi.label(hot)
    mask = np.zeros(flair.shape, np.uint8)
    if n == 0:
        return mask
    sizes = ndi.sum(hot, lab, range(1, n + 1))
    whole = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    enh = whole & (t1ce > 1100)
    necro = whole & (t1ce < 350)
    mask[whole] = 1
    mask[necro] = 2
    mask[enh] = 4
    return mask


class InferenceService:
    def __init__(self):
        self.predictor = registry.get(settings.active_predictor)
        weights_path = settings.model_weights_path or (PROJECT_ROOT / "runs/exp/best.pt")
        if weights_path.exists():
            self.predictor.load_weights(weights_path)
            print(f"Loaded model weights from {weights_path}")
        else:
            print(f"WARNING: weights not found at {weights_path}, running UNTRAINED model")

    def _load_volume_from_dir(self, case_dir: Path) -> VolumeInput:
        """
        Load modalities from a directory.
        Expects files with suffixes like _t1.nii.gz, _t1ce.nii.gz, etc.
        """
        ordered_modalities: list[VolumeModality] = []
        arrays = []
        spacing: tuple[float, float, float] | None = None
        affine: np.ndarray | None = None

        target_modalities = [VolumeModality.T1CE, VolumeModality.T2, VolumeModality.FLAIR]

        for mod in target_modalities:
            found = False
            for f in case_dir.iterdir():
                stem = f.name.replace(".nii.gz", "").replace(".nii", "")
                if stem.endswith(f"_{mod.value}"):
                    img = nib.load(str(f))
                    arr = img.get_fdata().astype(np.float32)
                    arrays.append(arr)
                    if spacing is None:
                        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
                    if affine is None:
                        affine = img.affine
                    ordered_modalities.append(mod)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Missing required modality '{mod.value}' in {case_dir}. "
                    f"Expected a file ending in '_{mod.value}.nii' or '_{mod.value}.nii.gz'."
                )

        stacked = np.stack(arrays, axis=0)
        self._raw = stacked  # (3, D, H, W): T1ce, T2, FLAIR
        return VolumeInput(
            modalities=ordered_modalities,
            array=stacked,
            spacing=spacing,
            affine=affine,
        )

    def run_inference(self, case_id: int, case_dir: Path) -> Prediction:
        try:
            return self._run_inference(case_id, case_dir)
        except Exception:
            import traceback

            traceback.print_exc()
            raise

    def _run_inference(self, case_id: int, case_dir: Path) -> Prediction:
        # 1. Load volume
        volume = self._load_volume_from_dir(case_dir)

        # 2. Run the full segmentation -> postprocess -> features pipeline.
        #    MC-Dropout uncertainty is skipped here to keep this basic/fast;
        #    flip compute_uncertainty=True if you want it later.
        if DEMO_MODE:
            # Skip the (slow, untrained) neural network entirely.
            print("WARNING: DEMO_MODE on, skipping the model and using a fake threshold mask")
            output = SimpleNamespace(
                mask=None, uncertainty=None, saliency=None,
                meta=SimpleNamespace(inference_seconds=0.0),
            )
            mask = _demo_mask(self._raw)
            # Real feature extraction on the demo mask so the dashboard renders normally.
            feature_bundle = extract_features(mask=mask, volume=volume)
            print(f"DEBUG demo mask shape={mask.shape}, tumor voxels={int((mask > 0).sum())}")
        else:
            # The network was trained on volumes that went through the
            # validation transforms (RAS -> 1 mm -> percentile scale -> z-score).
            # Feeding raw scanner intensities gives meaningless masks, so apply
            # the same preprocessing here and map the result back afterwards.
            model_array, model_affine, prep = prepare_model_input(
                np.asarray(volume.array),
                volume.affine,
                volume.spacing,
            )
            model_volume = VolumeInput(
                modalities=volume.modalities,
                array=model_array,
                spacing=(1.0, 1.0, 1.0),
                affine=model_affine,
            )

            pipeline_output = run_pipeline(
                self.predictor,
                model_volume,
                postprocess_mask=True,
                compute_uncertainty=False,
            )
            output = pipeline_output.prediction
            feature_bundle = pipeline_output.features

            # Bring the mask back onto the uploaded scan's grid so previews,
            # Dice and the risk map line up with the original images.
            mask = restore_mask(output.mask, prep)
            print(
                f"DEBUG mask shape={mask.shape}, "
                f"tumor voxels={int((mask > 0).sum())}, "
                f"input shape={self._raw.shape[1:]}"
            )

        # 3. Save mask (and uncertainty/saliency if present) to disk.
        case_results_dir = settings.results_dir / f"case_{case_id}"
        case_results_dir.mkdir(parents=True, exist_ok=True)

        mask_path = case_results_dir / "mask.nii.gz"
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), volume.affine), str(mask_path))

        uncertainty_path = None
        if output.uncertainty is not None:
            uncertainty_path = case_results_dir / "uncertainty.nii.gz"
            nib.save(nib.Nifti1Image(output.uncertainty.astype(np.float32), volume.affine), str(uncertainty_path))

        saliency_path = None
        if output.saliency is not None:
            saliency_path = case_results_dir / "saliency.nii.gz"
            nib.save(nib.Nifti1Image(output.saliency.astype(np.float32), volume.affine), str(saliency_path))

        # 4. Heuristic surgical risk mapping (whole-tumor region vs. approximate
        #    functional regions — see braintumor.risk.mapping for the caveats).
        risk_regions = compute_risk_regions(
            tumor_mask=mask,
            volume_array=np.asarray(volume.array),
            spacing=volume.spacing,
        )
        wt_feature = next((f for f in feature_bundle.features if f.label.value == "whole_tumor"), None)
        tumor_volume_mm3 = wt_feature.volume_mm3 if wt_feature else None
        if DEMO_MODE:
            tumor_volume_mm3 = float((mask > 0).sum() * np.prod(volume.spacing))

        if not bool((mask > 0).any()):
            # Empty mask: report volume 0 and let build_report_text produce the
            # explicit "no tumor detected" report (never "all regions Low").
            tumor_volume_mm3 = 0.0

        risk_report_text = build_report_text(
            case_name=f"case_{case_id}",
            regions=risk_regions,
            tumor_volume_mm3=tumor_volume_mm3,
        )
        risk_regions_json = [
            {
                "region": r.region,
                "function": r.function,
                "overlap_pct": r.overlap_pct,
                "distance_mm": r.distance_mm,
                "risk_tier": r.risk_tier,
            }
            for r in risk_regions
        ]

        # 5. Store in DB
        with get_session() as session:
            prediction = Prediction(
                case_id=case_id,
                model_name=self.predictor.name,
                model_version=self.predictor.version,
                inference_seconds=output.meta.inference_seconds,
                mask_path=str(mask_path),
                uncertainty_path=str(uncertainty_path) if uncertainty_path else None,
                saliency_path=str(saliency_path) if saliency_path else None,
                mesh_paths={},
                features=feature_bundle.model_dump(),
                metrics=None,
                risk_regions=risk_regions_json,
                risk_report=risk_report_text,
            )
            session.add(prediction)
            session.commit()
            session.refresh(prediction)
            return prediction
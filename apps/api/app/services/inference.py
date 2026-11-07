
"""Service for running ML inference on MRI volumes."""

from __future__ import annotations

import numpy as np
import nibabel as nib
from pathlib import Path

from monai.data import MetaTensor

from app.config import settings
from app.db import Prediction, get_session
from braintumor.registry import registry
from braintumor.schemas.contracts import VolumeInput, VolumeModality
from braintumor.data.preprocessing import val_transforms
from braintumor.inference.pipeline import run_pipeline
from braintumor.risk import compute_risk_regions, build_report_text


# Register the predictor once.
if "attn_unet_2d" not in registry.list_names():
    from braintumor.models.attention_unet_2d import build_attn_unet_2d

    registry.register("attn_unet_2d", build_attn_unet_2d)

if "unet_3d" not in registry.list_names():
    from braintumor.models.unet_3d import build_unet_3d

    registry.register("unet_3d", build_unet_3d)


SUPPORTED_MODELS: tuple[str, ...] = ("attn_unet_2d", "unet_3d")


class InferenceService:
    def __init__(self):
        self._predictors: dict[str, Any] = {}

    @property
    def predictor(self):
        """Backward-compatible default predictor property."""
        return self.get_predictor(settings.active_predictor)

    def get_predictor(self, model_name: str | None = None) -> Any:
        """Resolve and load the requested model with its independent checkpoint.

        Parameters
        ----------
        model_name : str, optional
            'attn_unet_2d' or 'unet_3d'. Defaults to settings.active_predictor ('attn_unet_2d').

        Returns
        -------
        Predictor instance loaded with its dedicated weights.

        Raises
        ------
        ValueError: If model_name is unsupported.
        FileNotFoundError: If the requested checkpoint does not exist (no silent fallback).
        """
        name = model_name or settings.active_predictor
        if name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_name '{name}'. Supported models: {list(SUPPORTED_MODELS)}"
            )

        if name in self._predictors:
            return self._predictors[name]

        predictor = registry.get(name)

        if name == "attn_unet_2d":
            weights_path = settings.model_weights_path or Path("runs/exp/best.pt")
        elif name == "unet_3d":
            weights_path = (
                getattr(settings, "model_weights_path_3d", None)
                or (Path("runs/exp_3d_v2/best.pt") if Path("runs/exp_3d_v2/best.pt").exists() else Path("runs/exp_3d/best.pt"))
            )
        else:
            raise ValueError(f"Unknown model: {name}")

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for model '{name}' not found at {weights_path}. "
                "Ensure the model checkpoint exists before triggering inference."
            )

        predictor.load_weights(weights_path)
        self._predictors[name] = predictor
        return predictor

    def _load_volume_from_dir(self, case_dir: Path) -> VolumeInput:
        """Load and preprocess MRI modalities using validation transforms.

        Internal array layout:
            (C, D, H, W)

        The stored affine remains the conventional RAS (X,Y,Z) affine.
        """

        target_modalities = [
            VolumeModality.T1CE,
            VolumeModality.T2,
            VolumeModality.FLAIR,
        ]

        input_data = {}

        for modality in target_modalities:
            suffix = f"_{modality.value}"

            matching_files = [
                f for f in case_dir.iterdir()
                if f.is_file()
                and (
                    f.name.endswith(suffix + ".nii.gz")
                    or f.name.endswith(suffix + ".nii")
                )
            ]

            if not matching_files:
                raise ValueError(
                    f"Missing required modality '{modality.value}' "
                    f"in {case_dir}"
                )

            if len(matching_files) > 1:
                raise ValueError(
                    f"Multiple files found for modality "
                    f"'{modality.value}' in {case_dir}: {matching_files}"
                )

            input_data[modality.value] = str(matching_files[0])

        # The empty keys list tells the transform pipeline that no
        # segmentation label is being supplied.
        transform = val_transforms(
            target_modalities,
            keys=[],
        )

        processed = transform(input_data)

        arrays = []
        ras_affine = None
        expected_shape = None

        for modality in target_modalities:
            image = processed[modality.value]

            # MONAI output: (1, X, Y, Z).
            image_array = image.detach().cpu().numpy()

            if image_array.ndim != 4 or image_array.shape[0] != 1:
                raise ValueError(
                    f"Unexpected transformed shape for "
                    f"{modality.value}: {image_array.shape}"
                )

            current_affine = np.asarray(
                image.affine,
                dtype=np.float64,
            )

            if ras_affine is None:
                ras_affine = current_affine
                expected_shape = image_array.shape[1:]
            else:
                if image_array.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Transformed shape mismatch for {modality.value}"
                    )

                if not np.allclose(
                    current_affine,
                    ras_affine,
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"Transformed affine mismatch for {modality.value}"
                    )

            # Convert (X,Y,Z) into the predictor's (D,H,W) order.
            # D=Z, H=X, W=Y.
            volume_dhw = np.transpose(
                image_array[0],
                (2, 0, 1),
            )

            arrays.append(volume_dhw.astype(np.float32))

        if ras_affine is None:
            raise ValueError("Could not determine transformed RAS affine")

        # RAS affine voxel spacing is in (X,Y,Z) order.
        xyz_spacing = np.linalg.norm(
            ras_affine[:3, :3],
            axis=0,
        )

        # VolumeInput spacing must follow (D,H,W) = (Z,X,Y).
        dhw_spacing = (
            float(xyz_spacing[2]),
            float(xyz_spacing[0]),
            float(xyz_spacing[1]),
        )

        stacked = np.stack(arrays, axis=0)

        return VolumeInput(
            modalities=target_modalities,
            array=stacked,
            spacing=dhw_spacing,
            affine=ras_affine,
        )

    @staticmethod
    def _save_scalar_map(
        data: np.ndarray,
        path: Path,
        affine: np.ndarray,
        dtype,
    ) -> None:
        """Save a (D,H,W) map as a RAS-oriented (X,Y,Z) NIfTI."""
        data_xyz = np.transpose(
            np.asarray(data),
            (1, 2, 0),
        )

        image = nib.Nifti1Image(
            data_xyz.astype(dtype),
            affine,
        )

        nib.save(image, str(path))

    def run_inference(
        self,
        case_id: int,
        case_dir: Path,
        model_name: str | None = None,
    ) -> Prediction:
        predictor = self.get_predictor(model_name)

        # 1. Load and preprocess MRI.
        volume = self._load_volume_from_dir(case_dir)

        # 2. Run segmentation, postprocessing and feature extraction.
        pipeline_output = run_pipeline(
            predictor,
            volume,
            postprocess_mask=True,
            compute_uncertainty=False,
        )

        output = pipeline_output.prediction
        feature_bundle = pipeline_output.features

        # 3. Save output maps.
        case_results_dir = settings.results_dir / f"case_{case_id}"
        case_results_dir.mkdir(parents=True, exist_ok=True)

        mask_path = case_results_dir / "mask.nii.gz"

        self._save_scalar_map(
            output.mask,
            mask_path,
            volume.affine,
            np.uint8,
        )

        uncertainty_path = None

        if output.uncertainty is not None:
            uncertainty_path = case_results_dir / "uncertainty.nii.gz"

            self._save_scalar_map(
                output.uncertainty,
                uncertainty_path,
                volume.affine,
                np.float32,
            )

        saliency_path = None

        if output.saliency is not None:
            saliency_path = case_results_dir / "saliency.nii.gz"

            self._save_scalar_map(
                output.saliency,
                saliency_path,
                volume.affine,
                np.float32,
            )

        # 4. Atlas-based anatomical surgical risk mapping (Harvard-Oxford atlas).
        risk_regions = compute_risk_regions(
            tumor_mask=output.mask,
            volume_array=np.asarray(volume.array),
            spacing=volume.spacing,
            affine=volume.affine,
        )

        wt_feature = next(
            (
                f for f in feature_bundle.features
                if f.label.value == "whole_tumor"
            ),
            None,
        )

        tumor_volume_mm3 = (
            wt_feature.volume_mm3 if wt_feature else None
        )

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

        # 5. Store results in the database.
        with get_session() as session:
            prediction = Prediction(
                case_id=case_id,
                model_name=predictor.name,
                model_version=predictor.version,
                inference_seconds=output.meta.inference_seconds,
                mask_path=str(mask_path),
                uncertainty_path=(
                    str(uncertainty_path)
                    if uncertainty_path else None
                ),
                saliency_path=(
                    str(saliency_path)
                    if saliency_path else None
                ),
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

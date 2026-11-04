"""Verification script for end-to-end inference.
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from braintumor.config import config
from braintumor.registry import registry
from braintumor.data.brats_loader import BratsCase, discover_cases
from braintumor.schemas.contracts import VolumeInput, VolumeModality

def verify_inference():
    # 1. Load Predictor
    from braintumor.models.attention_unet_2d import build_attn_unet_2d
    from braintumor.registry import registry
    registry.register("attn_unet_2d", build_attn_unet_2d)

    predictor = registry.get(config.active_predictor)
    ckpt_path = Path("runs/one_epoch_test/best.pt")
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return

    predictor.load_weights(ckpt_path)
    print(f"Loaded model: {predictor.name} v{predictor.version}")

    # 2. Get a real (synthetic) sample
    cases = discover_cases(config.data_root)
    if not cases:
        print("No cases found!")
        return
    case = cases[0]
    print(f"Testing inference on case: {case.case_id}")

    # 3. Prepare VolumeInput
    # We need to load the data manually for the Predictor
    import nibabel as nib
    modalities = [VolumeModality.T1CE, VolumeModality.T2, VolumeModality.FLAIR]
    arrays = []
    for mod in modalities:
        # Find the file that ends with the modality name
        mod_file = next(f for f in case.root.iterdir() if f.name.endswith(f"_{mod.value}.nii.gz"))
        img = nib.load(str(mod_file))
        arrays.append(img.get_fdata().astype(np.float32))

    volume = VolumeInput(
        modalities=modalities,
        array=np.stack(arrays, axis=0).transpose(0, 3, 1, 2),
        spacing=(1.0, 1.0, 1.0),
        affine=np.eye(4),
        case_id=case.case_id,
        patient_id=case.patient_id
    )

    # 4. Predict
    output = predictor.predict(volume)
    print("Prediction successful!")
    print(f"Mask shape: {output.mask.shape}")
    print(f"Inference time: {output.meta.inference_seconds:.4f}s")

if __name__ == "__main__":
    verify_inference()

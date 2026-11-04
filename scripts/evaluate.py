
"""Full-volume evaluation of the BraTS 2D segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from braintumor.config import config
from braintumor.data.dataset import BratsDataset
from braintumor.data.preprocessing import (
    val_transforms,
    fourclass_to_brats,
)
from braintumor.data.splits import discover_and_split
from braintumor.eval.metrics import evaluate_case, aggregate_metrics
from braintumor.models.attention_unet_2d import (
    MVP_MODALITIES,
    build_attn_unet_2d,
)
from braintumor.registry import registry
from braintumor.inference.postprocess import postprocess


def get_volume_spacing(label) -> tuple[float, float, float]:
    """Get physical spacing from the transformed MONAI volume."""

    affine = getattr(label, "affine", None)

    if affine is not None:
        affine = np.asarray(affine.cpu() if torch.is_tensor(affine) else affine)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)

        if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
            return tuple(float(v) for v in spacing)

    # val_transforms applies Spacingd(pixdim=(1, 1, 1)).
    return (1.0, 1.0, 1.0)


def evaluate_patient(
    model,
    sample: dict,
    modalities,
    device: str,
    slice_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:

    # Assemble MRI channels in exactly the model's expected order.
    channels = []

    for modality in modalities:
        key = modality.value
        channel = torch.as_tensor(sample[key]).float()

        if channel.ndim == 4 and channel.shape[0] == 1:
            channel = channel.squeeze(0)

        if channel.ndim != 3:
            raise ValueError(
                f"Expected {key} to have shape (D,H,W), "
                f"got {tuple(channel.shape)}"
            )

        channels.append(channel)

    image = torch.stack(channels, dim=0)

    # Ground-truth labels remain in native BraTS format.
    label = torch.as_tensor(sample["label"])

    if label.ndim == 4 and label.shape[0] == 1:
        label = label.squeeze(0)

    if label.ndim != 3:
        raise ValueError(
            f"Expected label shape (D,H,W), got {tuple(label.shape)}"
        )

    if tuple(image.shape[1:]) != tuple(label.shape):
        raise ValueError(
            f"Image shape {tuple(image.shape[1:])} "
            f"does not match label shape {tuple(label.shape)}"
        )

    spacing = get_volume_spacing(sample["label"])

    # The last spatial axis contains the slices used by our 2D model.
    number_of_slices = image.shape[-1]
    predicted_slices = []

    model.eval()

    with torch.no_grad():
        for start in range(0, number_of_slices, slice_batch_size):
            end = min(start + slice_batch_size, number_of_slices)

            # Each slice has shape (C,D,H).
            batch_slices = [
                image[..., z]
                for z in range(start, end)
            ]

            batch = torch.stack(batch_slices, dim=0).to(device)

            logits = model(batch)
            predictions = torch.argmax(logits, dim=1)

            predicted_slices.extend(
                predictions.cpu().numpy()
            )

    # Reconstruct the complete 3D volume: (D,H,W).
    prediction_4class = np.stack(predicted_slices, axis=-1)

    # Convert model classes {0,1,2,3} to native BraTS {0,1,2,4}.
    prediction_native = fourclass_to_brats(prediction_4class)

    # Ground truth is already native BraTS {0,1,2,4}.
    target_native = label.cpu().numpy().astype(np.uint8)

    if prediction_native.shape != target_native.shape:
        raise ValueError(
            f"Prediction shape {prediction_native.shape} "
            f"does not match target shape {target_native.shape}"
        )

    return prediction_native, target_native, spacing


def main():
    parser = argparse.ArgumentParser(
        description="Full-volume BraTS model evaluation"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=str(config.data_root),
    )

    parser.add_argument(
        "--split_file",
        type=str,
        default=str(config.split_file),
    )

    parser.add_argument(
        "--slice_batch_size",
        type=int,
        default=8,
        help="Number of 2D slices processed together",
    )

    args = parser.parse_args()

    if args.slice_batch_size < 1:
        parser.error("--slice_batch_size must be at least 1")

    # 1. Load the patient-level test split.
    root = Path(args.data_root)

    train, val, test = discover_and_split(
        root,
        split_file=Path(args.split_file),
    )

    if not test:
        raise RuntimeError("The test split contains no patients.")

    modalities = list(MVP_MODALITIES)

    test_ds = BratsDataset(
        cases=test,
        modalities=modalities,
        transform=val_transforms(modalities),
    )

    # 2. Load the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluation device: {device}")

    registry.register("attn_unet_2d", build_attn_unet_2d)

    model = registry.get("attn_unet_2d").model.to(device)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Evaluate each complete patient volume.
    print(f"\nEvaluating {len(test)} complete test volumes...")

    raw_metrics_all = []
    processed_metrics_all = []

    for i in range(len(test_ds)):
        case = test[i]
        sample = test_ds[i]

        prediction, target, spacing = evaluate_patient(
            model=model,
            sample=sample,
            modalities=modalities,
            device=device,
            slice_batch_size=args.slice_batch_size,
        )

        raw_metrics = evaluate_case(prediction, target, spacing)

        processed_prediction = postprocess(prediction)

        processed_metrics = evaluate_case(
            processed_prediction,
            target,
            spacing,
        )

        raw_metrics_all.append(raw_metrics)
        processed_metrics_all.append(processed_metrics)

        print(f"\n[{i + 1}/{len(test_ds)}] {case.patient_id}")

        for region in ("WT", "TC", "ET"):
            print(
                f"  {region} | "
                f"Raw Dice: {raw_metrics[region].dice:.4f} | "
                f"Postprocessed Dice: {processed_metrics[region].dice:.4f}"
            )

    raw_aggregated = aggregate_metrics(raw_metrics_all)
    processed_aggregated = aggregate_metrics(processed_metrics_all)

    def print_metrics(title, aggregated):
        print(f"\n{'=' * 65}")
        print(title)
        print(f"{'=' * 65}")

        for region, metrics in aggregated.items():
            print(f"\nRegion: {region}")

            for metric, values in metrics.items():
                print(
                    f"  {metric}: "
                    f"{values['mean']:.4f} +/- "
                    f"{values['std']:.4f} "
                    f"(n={values['n']})"
                )

    print_metrics("RAW PREDICTION METRICS", raw_aggregated)
    print_metrics("POSTPROCESSED PREDICTION METRICS", processed_aggregated)

    print("\nRAW VS POSTPROCESSED EVALUATION COMPLETED.")

if __name__ == "__main__":
    main()


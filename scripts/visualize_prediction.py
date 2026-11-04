
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from braintumor.config import config
from braintumor.registry import registry
from braintumor.data.brats_loader import discover_cases
from braintumor.schemas.contracts import VolumeInput, VolumeModality
from braintumor.models.attention_unet_2d import build_attn_unet_2d


def print_class_counts(title, mask):
    print(f"\n{title}")

    labels, counts = np.unique(mask, return_counts=True)

    class_names = {
        0: "Background",
        1: "NCR/NET",
        2: "Edema",
        4: "Enhancing Tumour",
    }

    for label, count in zip(labels, counts):
        name = class_names.get(int(label), "Unknown")
        print(f"Label {int(label)} ({name}): {int(count)} voxels")


def visualize_prediction():

    # 1. Load model
    registry.register("attn_unet_2d", build_attn_unet_2d)

    predictor = registry.get(config.active_predictor)

    checkpoint = Path("runs/one_epoch_test/best.pt")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    predictor.load_weights(checkpoint)

    print("Model loaded successfully!")

    # 2. Load BraTS case
    cases = discover_cases(config.data_root)

    if not cases:
        raise RuntimeError("No BraTS cases found!")

    case = cases[0]

    print("Case:", case.case_id)

    modalities = [
        VolumeModality.T1CE,
        VolumeModality.T2,
        VolumeModality.FLAIR,
    ]

    arrays = []

    for mod in modalities:

        file = next(
            f for f in case.root.iterdir()
            if f.name.endswith(f"_{mod.value}.nii.gz")
        )

        img = nib.load(str(file))

        arrays.append(img.get_fdata().astype(np.float32))

    # Convert (C, X, Y, Z) to (C, Z, X, Y)
    volume_array = np.stack(arrays, axis=0).transpose(0, 3, 1, 2)

    # 3. Prepare model input
    volume = VolumeInput(
        modalities=modalities,
        array=volume_array,
        spacing=(1.0, 1.0, 1.0),
        affine=np.eye(4),
        case_id=case.case_id,
        patient_id=case.patient_id,
    )

    # 4. Generate prediction
    print("\nGenerating segmentation...")

    output = predictor.predict(volume)

    predicted_mask = output.mask

    print("\nPrediction completed!")
    print("Prediction mask shape:", predicted_mask.shape)

    # 5. Load ground truth
    seg_file = next(
        f for f in case.root.iterdir()
        if f.name.endswith("_seg.nii.gz")
    )

    ground_truth = nib.load(str(seg_file)).get_fdata()

    print("Ground truth shape:", ground_truth.shape)

    # 6. Print class counts
    print_class_counts("PREDICTED CLASS COUNTS", predicted_mask)

    print_class_counts("GROUND TRUTH CLASS COUNTS", ground_truth)

    # 7. Find slice with largest ground-truth tumour
    ground_truth_counts = np.count_nonzero(
        ground_truth,
        axis=(0, 1)
    )

    slice_index = int(np.argmax(ground_truth_counts))

    print("\nSelected slice:", slice_index)

    print(
        "Ground-truth tumour voxels in selected slice:",
        int(ground_truth_counts[slice_index])
    )

    print(
        "Predicted tumour voxels in selected slice:",
        int(np.count_nonzero(predicted_mask[slice_index]))
    )

    # 8. Extract MRI slices
    t1ce_slice = volume_array[0, slice_index]

    gt_slice = ground_truth[:, :, slice_index]

    pred_slice = predicted_mask[slice_index]

    # 9. Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].imshow(t1ce_slice, cmap="gray")
    axes[0].set_title("Original T1ce MRI")

    axes[1].imshow(t1ce_slice, cmap="gray")
    axes[1].set_title("Predicted Tumour Segmentation")

    axes[2].imshow(t1ce_slice, cmap="gray")
    axes[2].set_title("BraTS Ground Truth")

    labels = [
        (1, "red", "NCR/NET"),
        (2, "lime", "Edema"),
        (4, "yellow", "Enhancing Tumour"),
    ]

    for label, color, name in labels:

        if np.any(pred_slice == label):
            axes[1].contour(
                pred_slice == label,
                levels=[0.5],
                colors=[color],
                linewidths=1,
            )

        if np.any(gt_slice == label):
            axes[2].contour(
                gt_slice == label,
                levels=[0.5],
                colors=[color],
                linewidths=1,
            )

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"BraTS Case: {case.case_id} | Slice: {slice_index}"
    )

    plt.tight_layout()

    # 10. Save visualization
    output_dir = Path("outputs/visualizations")

    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "prediction_visualization.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    print("\nVisualization saved at:", save_path)

    plt.show()


if __name__ == "__main__":
    visualize_prediction()

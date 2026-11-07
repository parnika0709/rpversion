"""Independent training script for the Experimental Volumetric 3D U-Net.

Developed in response to Reviewer Comment 7.
- Operates independently from the 2D Attention U-Net training pipeline.
- Uses genuine 3D volumetric patches with 3D convolutions and skip connections.
- Uses identical patient-level data splits to prevent data leakage.
- Preserves the existing 2D Attention U-Net baseline untouched.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader, Dataset
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

from braintumor.config import config
from braintumor.data.brats_loader import BratsCase
from braintumor.data.preprocessing import brats_label_to_4class
from braintumor.data.splits import discover_and_split
from braintumor.data.synthetic import setup_synthetic_dataset
from braintumor.models.attention_unet_2d import MVP_MODALITIES
from braintumor.models.unet_3d import Unet3D
from braintumor.training.losses import ClassWeightedDiceCELoss, DiceCELoss


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvertBraTSLabelsd:
    """Dictionary transform to map BraTS labels {0, 1, 2, 4} to {0, 1, 2, 3}."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = brats_label_to_4class(d[key])
        return d


def build_3d_train_transforms(
    patch_size: tuple[int, int, int],
    pos: float = 1.0,
    neg: float = 1.0,
    num_samples: int = 4,
):
    """Augmented volumetric 3D patch training transforms with balanced pos/neg sampling."""
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1.0,
            upper=99.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ConvertBraTSLabelsd(keys=("label",)),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=patch_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_key=None,
        ),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        EnsureTyped(keys=keys),
        # Cast to plain torch Tensor to avoid MetaTensor metadata conflicts
        # during loss computation and batch collation.
        CastToTyped(keys=["image"], dtype=torch.float32),
        CastToTyped(keys=["label"], dtype=torch.long),
    ])


def build_3d_val_transforms(patch_size: tuple[int, int, int]):
    """Deterministic validation volumetric transforms."""
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1.0,
            upper=99.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ConvertBraTSLabelsd(keys=("label",)),
        RandSpatialCropd(keys=keys, roi_size=patch_size, random_size=False),
        EnsureTyped(keys=keys),
        # Cast to plain torch Tensor to avoid MetaTensor metadata conflicts
        # during loss computation and batch collation.
        CastToTyped(keys=["image"], dtype=torch.float32),
        CastToTyped(keys=["label"], dtype=torch.long),
    ])


def cases_to_monai_dicts(cases: list[BratsCase], modalities: list) -> list[dict]:
    """Convert BratsCase list to MONAI dictionary items combining modalities into a single list."""
    items = []
    for c in cases:
        img_paths = [str(c.modality_paths[m]) for m in modalities if m in c.modality_paths]
        if len(img_paths) != len(modalities):
            continue
        item = {"image": img_paths}
        if c.seg_path:
            item["label"] = str(c.seg_path)
            items.append(item)
    return items


def main():
    parser = argparse.ArgumentParser(description="Train Volumetric 3D U-Net Model")
    parser.add_argument("--data_root", type=str, default=str(config.data_root))
    parser.add_argument("--split_file", type=str, default=str(config.split_file))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch_size", type=int, default=64, help="Cubic patch size (e.g. 64 for 64x64x64)")
    parser.add_argument("--samples_per_volume", type=int, default=4, help="Number of 3D crops per volume")
    parser.add_argument("--save_dir", type=str, default="runs/exp_3d_v2")
    parser.add_argument("--imbalance_aware", action="store_true", help="Use class-weighted Dice+CE loss with balanced background")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for pipeline verification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs")

    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        # In synthetic mode use a local temporary directory so the script works
        # out-of-the-box without real BraTS data, even when config.data_root
        # points to a machine-specific or non-existent Windows path.
        import tempfile
        tmp_root = Path(tempfile.mkdtemp(prefix="neurorisk_syn_"))
        print(f"Using synthetic dataset for 3D verification (temp dir: {tmp_root})...")
        setup_synthetic_dataset(tmp_root, n_cases=8)
        # Do NOT use the stale split_file from config — generate a fresh split.
        train_cases, val_cases, test_cases = discover_and_split(tmp_root, split_file=None)
    else:
        root = Path(args.data_root)
        split_path = Path(args.split_file) if args.split_file and Path(args.split_file).is_file() else None
        train_cases, val_cases, test_cases = discover_and_split(root, split_file=split_path)
    print(f"\n3D Volumetric Training Split:")
    print(f"  Training patients:   {len(train_cases)}")
    print(f"  Validation patients: {len(val_cases)}")
    print(f"  Test patients:       {len(test_cases)}")

    modalities = list(MVP_MODALITIES)
    patch_size = (args.patch_size, args.patch_size, args.patch_size)

    train_dicts = cases_to_monai_dicts(train_cases, modalities)
    val_dicts = cases_to_monai_dicts(val_cases, modalities)

    if not train_dicts:
        raise RuntimeError(
            f"No training cases with all required modalities {[m.value for m in modalities]} were found. "
            "Check data_root or use --synthetic for a self-contained pipeline test."
        )
    if not val_dicts:
        # Fall back to using 1 training sample as val for smoke-test purposes
        print("WARNING: No validation cases found — using 1 training case as validation for smoke test.")
        val_dicts = train_dicts[:1]

    train_ds = Dataset(
        data=train_dicts,
        transform=build_3d_train_transforms(
            patch_size,
            pos=1.0,
            neg=1.0,
            num_samples=args.samples_per_volume,
        ),
    )
    val_ds = Dataset(data=val_dicts, transform=build_3d_val_transforms(patch_size))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_list_data_collate,
    )

    # 3D Model initialization
    model = Unet3D(
        in_channels=3,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Imbalance-aware loss with balanced background weighting (CE weight 1.0 for background, not 0.1)
    if args.imbalance_aware:
        print("Using ClassWeightedDiceCELoss with balanced background penalty...")
        loss_fn = ClassWeightedDiceCELoss(
            dice_weight=0.5,
            ce_weight=0.5,
            include_background=False,
            dice_class_weights=(1.5, 1.0, 1.5),
            ce_class_weights=(1.0, 2.0, 1.5, 2.0),
        ).to(device)
    else:
        loss_fn = DiceCELoss(dice_weight=0.5, ce_weight=0.5).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    no_improve_epochs = 0
    print(f"\nStarting 3D model training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            # Strip MetaTensor metadata to plain torch.Tensor before loss computation.
            if hasattr(x, "as_tensor"):
                x = x.as_tensor()
            # label shape: (B, 1, D, H, W) -> squeeze to (B, D, H, W)
            y = batch["label"].to(device)
            if hasattr(y, "as_tensor"):
                y = y.as_tensor()
            if y.ndim == 5 and y.shape[1] == 1:
                y = y.squeeze(1)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += float(loss.item())
            batches += 1

        avg_train_loss = train_loss / max(1, batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                if hasattr(x, "as_tensor"):
                    x = x.as_tensor()
                y = batch["label"].to(device)
                if hasattr(y, "as_tensor"):
                    y = y.as_tensor()
                if y.ndim == 5 and y.shape[1] == 1:
                    y = y.squeeze(1)

                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += float(loss.item())
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "model_name": "unet_3d",
                "patch_size": patch_size,
            },
            save_dir / "last.pt",
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "model_name": "unet_3d",
                    "patch_size": patch_size,
                },
                save_dir / "best.pt",
            )
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs without improvement.")
                break

    print(f"\n3D Training finished. Best Val Loss: {best_val_loss:.4f}. Saved to {save_dir}")


if __name__ == "__main__":
    main()

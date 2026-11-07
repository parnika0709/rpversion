
"""Main training script for the Brain Tumor segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from braintumor.config import config
from braintumor.data.dataset import BratsDataset
from braintumor.data.slicing import SlicingDataset
from braintumor.data.preprocessing import train_transforms, val_transforms
from braintumor.data.splits import discover_and_split
from braintumor.data.synthetic import setup_synthetic_dataset
from braintumor.models.attention_unet_2d import (
    AttentionUnet2D,
    MVP_MODALITIES,
)
from braintumor.training.train import fit, TrainConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train Brain Tumor Segmentation Model"
    )

    parser.add_argument("--data_root", type=str, default=str(config.data_root))
    parser.add_argument("--split_file", type=str, default=str(config.split_file))
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--lr", type=float, default=config.learning_rate)

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for verification",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be at least 1")

    # 1. Dataset setup
    root = Path(args.data_root)
    split_path = Path(args.split_file)

    if args.synthetic:
        print("Using synthetic dataset for verification...")
        setup_synthetic_dataset(root, n_cases=20)

    try:
        train_cases, val_cases, test_cases = discover_and_split(
            root,
            split_file=split_path,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Dataset setup failed: {error}")

    print("\nPatient-level dataset split:")
    print(f"Training patients:   {len(train_cases)}")
    print(f"Validation patients: {len(val_cases)}")
    print(f"Test patients:       {len(test_cases)}")

    if not train_cases or not val_cases or not test_cases:
        raise RuntimeError(
            "Training, validation, and test splits must all contain patients."
        )

    # 2. Modalities
    modalities = list(MVP_MODALITIES)

    print("\nMRI modalities:")
    print([modality.value for modality in modalities])

    # 3. Patient datasets
    train_ds = BratsDataset(
        train_cases,
        modalities,
        transform=train_transforms(modalities),
    )

    val_ds = BratsDataset(
        val_cases,
        modalities,
        transform=val_transforms(modalities),
    )

    # 4. Slice datasets
    train_slice_ds = SlicingDataset(
        train_ds,
        training=True,
        samples_per_volume=8,
        foreground_probability=0.7,
    )

    val_slice_ds = SlicingDataset(
        val_ds,
        training=False,
        slices_per_volume=155,
    )

    print("\nSlice dataset sizes:")
    print(f"Training slices:   {len(train_slice_ds)}")
    print(f"Validation slices: {len(val_slice_ds)}")

    if len(train_slice_ds) == 0 or len(val_slice_ds) == 0:
        raise RuntimeError("Training or validation slice dataset is empty.")

    # 5. Device
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Set USE_CUDA=false to use CPU."
        )

    device = config.device
    use_pin_memory = device.startswith("cuda")

    print("\nTraining device:", device)

    # 6. Data loaders
    train_loader = DataLoader(
        train_slice_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_slice_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )

    # 7. Model
    model = AttentionUnet2D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.model_channels,
        strides=config.model_strides,
        dropout=config.dropout,
    )

    # 8. Training configuration
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=config.weight_decay,
        seed=config.seed,
        save_dir=config.checkpoint_dir,
        num_workers=config.num_workers,
    )

    print("\nTraining configuration:")
    print("Epochs:", train_cfg.epochs)
    print("Batch size:", train_cfg.batch_size)
    print("Learning rate:", train_cfg.learning_rate)
    print("Weight decay:", train_cfg.weight_decay)
    print("Seed:", train_cfg.seed)
    print("Workers:", train_cfg.num_workers)
    print("Checkpoint directory:", train_cfg.save_dir)

    # 9. Training
    print("\nStarting model training...")

    history = fit(
        model,
        train_loader,
        val_loader,
        train_cfg,
        device=device,
    )

    print("\nTraining completed.")

    if history.get("val_loss"):
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

    print("Checkpoints directory:", train_cfg.save_dir)


if __name__ == "__main__":
    main()


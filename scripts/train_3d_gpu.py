"""Standalone GPU Training Script for Volumetric 3D U-Net.

Designed for high-throughput GPU training on CUDA hardware (>= 16-24 GB VRAM).
Features:
  - Full YAML configuration support via configs/train_3d_gpu.yaml with CLI overrides.
  - Standard clinical 3D patch resolution (e.g. 96x96x96 or 128x128x128).
  - Mixed precision training with torch.cuda.amp.autocast and GradScaler.
  - Multi-worker parallel data loading with pinned memory.
  - Gradient accumulation for effective batch sizing.
  - Class-weighted Dice + Cross-Entropy loss with balanced background.
  - Cosine annealing learning rate scheduling and early stopping.
  - Robust resumable training (--resume) with atomic checkpoint saving.
  - Separate best.pt and last.pt checkpoints preserving prior experiment directories.
  - Optional periodic whole-volume sliding-window validation.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from monai.data import DataLoader, Dataset
from monai.data.utils import pad_list_data_collate
from monai.inferers import sliding_window_inference
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
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)

from braintumor.config import config
from braintumor.data.brats_loader import BratsCase
from braintumor.data.preprocessing import brats_label_to_4class
from braintumor.data.splits import discover_and_split
from braintumor.models.attention_unet_2d import MVP_MODALITIES
from braintumor.models.unet_3d import Unet3D
from braintumor.training.losses import ClassWeightedDiceCELoss


PROTECTED_EXPERIMENT_DIRS: tuple[Path, ...] = (
    Path("runs/exp").resolve(),
    Path("runs/exp_3d").resolve(),
    Path("runs/exp_3d_v2").resolve(),
)


def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy and PyTorch CPU/CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvertBraTSLabelsd:
    """Transform BraTS native labels {0, 1, 2, 4} to contiguous classes {0, 1, 2, 3}."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = brats_label_to_4class(d[key])
        return d


def build_gpu_train_transforms(patch_size: tuple[int, int, int], samples_per_volume: int = 4):
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ConvertBraTSLabelsd(keys=("label",)),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=patch_size,
            pos=1.0,
            neg=1.0,
            num_samples=samples_per_volume,
            image_key=None,
        ),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        EnsureTyped(keys=keys),
        CastToTyped(keys=["image"], dtype=torch.float32),
        CastToTyped(keys=["label"], dtype=torch.long),
    ])


def build_gpu_val_transforms(patch_size: tuple[int, int, int]):
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ConvertBraTSLabelsd(keys=("label",)),
        RandSpatialCropd(keys=keys, roi_size=patch_size, random_size=False),
        EnsureTyped(keys=keys),
        CastToTyped(keys=["image"], dtype=torch.float32),
        CastToTyped(keys=["label"], dtype=torch.long),
    ])


def cases_to_monai_dicts(cases: list[BratsCase], modalities: list) -> list[dict]:
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


def atomic_save(state: dict[str, Any], target_path: Path) -> None:
    """Save checkpoint atomically to a temporary file before renaming to target."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.parent / f".{target_path.name}.tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, target_path)


def load_gpu_config(config_path: str | Path | None, cli_args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Load configuration from YAML file and merge with command-line overrides."""
    cfg: dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file '{path}' does not exist.")
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid YAML content in {path}; expected dictionary.")
            cfg = loaded

    # Validate top-level keys if YAML was provided
    known_sections = {"model", "data", "training", "loss", "inference"}
    for section in cfg:
        if section not in known_sections:
            raise ValueError(f"Unknown configuration section '{section}' in YAML. Supported sections: {list(known_sections)}")

    # Ensure structure sections exist with sensible defaults
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("inference", {})

    # Defaults for model
    cfg["model"].setdefault("name", "unet_3d")
    cfg["model"].setdefault("in_channels", 3)
    cfg["model"].setdefault("out_channels", 4)
    cfg["model"].setdefault("channels", [32, 64, 128, 256])
    cfg["model"].setdefault("strides", [2, 2, 2])
    cfg["model"].setdefault("num_res_units", 2)
    cfg["model"].setdefault("dropout", 0.2)

    # Defaults for data
    cfg["data"].setdefault("data_root", str(config.data_root))
    cfg["data"].setdefault("split_file", str(config.split_file))
    cfg["data"].setdefault("patch_size", [96, 96, 96])
    cfg["data"].setdefault("samples_per_volume", 4)
    cfg["data"].setdefault("num_workers", 4)
    cfg["data"].setdefault("pin_memory", True)

    # Defaults for training
    cfg["training"].setdefault("epochs", 100)
    cfg["training"].setdefault("batch_size", 2)
    cfg["training"].setdefault("gradient_accumulation_steps", 2)
    cfg["training"].setdefault("learning_rate", 2e-4)
    cfg["training"].setdefault("weight_decay", 1e-5)
    cfg["training"].setdefault("mixed_precision", True)
    cfg["training"].setdefault("patience", 15)
    cfg["training"].setdefault("seed", 42)
    cfg["training"].setdefault("save_dir", "runs/exp_3d_final")
    cfg["training"].setdefault("resume", None)
    cfg["training"].setdefault("val_sliding_window", False)
    cfg["training"].setdefault("val_interval_epochs", 5)

    # Defaults for loss
    cfg["loss"].setdefault("dice_weight", 0.5)
    cfg["loss"].setdefault("ce_weight", 0.5)
    cfg["loss"].setdefault("include_background", False)
    cfg["loss"].setdefault("dice_class_weights", [1.5, 1.0, 1.5])
    cfg["loss"].setdefault("ce_class_weights", [1.0, 2.0, 1.5, 2.0])

    # CLI Overrides take precedence if provided
    if cli_args is not None:
        if getattr(cli_args, "data_root", None) is not None:
            cfg["data"]["data_root"] = cli_args.data_root
        if getattr(cli_args, "split_file", None) is not None:
            cfg["data"]["split_file"] = cli_args.split_file
        if getattr(cli_args, "epochs", None) is not None:
            cfg["training"]["epochs"] = cli_args.epochs
        if getattr(cli_args, "batch_size", None) is not None:
            cfg["training"]["batch_size"] = cli_args.batch_size
        if getattr(cli_args, "lr", None) is not None:
            cfg["training"]["learning_rate"] = cli_args.lr
        if getattr(cli_args, "patch_size", None) is not None:
            p = cli_args.patch_size
            cfg["data"]["patch_size"] = [p, p, p] if isinstance(p, int) else list(p)
        if getattr(cli_args, "samples_per_volume", None) is not None:
            cfg["data"]["samples_per_volume"] = cli_args.samples_per_volume
        if getattr(cli_args, "save_dir", None) is not None:
            cfg["training"]["save_dir"] = cli_args.save_dir
        if getattr(cli_args, "resume", None) is not None:
            cfg["training"]["resume"] = cli_args.resume
        if getattr(cli_args, "num_workers", None) is not None:
            cfg["data"]["num_workers"] = cli_args.num_workers
        if getattr(cli_args, "grad_accum", None) is not None:
            cfg["training"]["gradient_accumulation_steps"] = cli_args.grad_accum
        if getattr(cli_args, "seed", None) is not None:
            cfg["training"]["seed"] = cli_args.seed
        if getattr(cli_args, "patience", None) is not None:
            cfg["training"]["patience"] = cli_args.patience
        if getattr(cli_args, "use_amp", None) is not None:
            cfg["training"]["mixed_precision"] = cli_args.use_amp
        if getattr(cli_args, "val_sliding_window", None) is not None:
            cfg["training"]["val_sliding_window"] = cli_args.val_sliding_window
        if getattr(cli_args, "val_interval_epochs", None) is not None:
            cfg["training"]["val_interval_epochs"] = cli_args.val_interval_epochs

    # Safety: Refuse to overwrite existing protected experiment checkpoints
    resolved_save_dir = Path(cfg["training"]["save_dir"]).resolve()
    for prot_dir in PROTECTED_EXPERIMENT_DIRS:
        if resolved_save_dir == prot_dir or prot_dir in resolved_save_dir.parents:
            raise ValueError(
                f"Refusing to write into protected experiment directory '{resolved_save_dir}'. "
                "Baseline 2D (runs/exp/) and prior 3D (runs/exp_3d/) checkpoints must remain preserved. "
                "Please specify an independent save_dir (e.g. 'runs/exp_3d_final')."
            )

    return cfg


def evaluate_sliding_window_validation(
    model: nn.Module,
    val_cases: list[BratsCase],
    modalities: list,
    roi_size: tuple[int, int, int],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
) -> dict[str, float]:
    """Calculate whole-volume Dice on validation cases using sliding-window inference.

    Notice: This provides internal experimental monitoring during training and does
    not constitute formal clinical validation.
    """
    model.eval()
    val_transforms_full = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ConvertBraTSLabelsd(keys=("label",)),
    ])

    wt_dices, tc_dices, et_dices = [], [], []

    def _binary_dice(p: np.ndarray, g: np.ndarray) -> float:
        inter = float(np.logical_and(p, g).sum())
        card = float(p.sum() + g.sum())
        return 1.0 if card == 0 else (2.0 * inter + 1e-5) / (card + 1e-5)

    with torch.no_grad():
        for case in val_cases:
            img_paths = [str(case.modality_paths[m]) for m in modalities if m in case.modality_paths]
            if len(img_paths) != len(modalities) or not case.seg_path:
                continue

            data = val_transforms_full({"image": img_paths, "label": str(case.seg_path)})
            x = data["image"].unsqueeze(0).to(device)
            y = np.asarray(data["label"].squeeze(0).cpu().numpy(), dtype=np.uint8)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = sliding_window_inference(
                    inputs=x,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )

            pred = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)

            # Class 0: BG, 1: NCR/NET, 2: ED, 3: ET
            wt_dices.append(_binary_dice(pred > 0, y > 0))
            tc_dices.append(_binary_dice(np.isin(pred, [1, 3]), np.isin(y, [1, 3])))
            et_dices.append(_binary_dice(pred == 3, y == 3))

    return {
        "val_wt_dice": float(np.mean(wt_dices)) if wt_dices else 0.0,
        "val_tc_dice": float(np.mean(tc_dices)) if tc_dices else 0.0,
        "val_et_dice": float(np.mean(et_dices)) if et_dices else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU Training Script for Volumetric 3D U-Net")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (e.g. configs/train_3d_gpu.yaml)")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory containing BraTS patient folders")
    parser.add_argument("--split_file", type=str, default=None, help="Path to patient split JSON")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patch_size", type=int, default=None, help="Volumetric patch size (e.g. 96 for 96x96x96)")
    parser.add_argument("--samples_per_volume", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None, help="Use FP16 mixed precision")
    parser.add_argument("--val_sliding_window", action=argparse.BooleanOptionalAction, default=None, help="Enable periodic whole-volume validation")
    parser.add_argument("--val_interval_epochs", type=int, default=None, help="Frequency (epochs) for whole-volume validation")

    cli_args = parser.parse_args()

    # 1. Load and resolve configuration
    config_file = cli_args.config or ("configs/train_3d_gpu.yaml" if Path("configs/train_3d_gpu.yaml").is_file() else None)
    cfg = load_gpu_config(config_file, cli_args)

    print("=" * 70)
    print("Resolved GPU Training Configuration:")
    print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    print("=" * 70)

    # 2. Strict GPU requirement check (No silent CPU fallback)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GPU training requires a CUDA-enabled GPU (>= 16-24 GB VRAM). "
            "CPU execution is not supported in train_3d_gpu.py."
        )

    device = torch.device("cuda")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    set_seed(cfg["training"]["seed"])

    # 3. Dataset directory existence check
    data_root = Path(cfg["data"]["data_root"])
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory '{data_root}' does not exist. "
            "Please ensure the BraTS dataset is mounted or specify --data_root."
        )

    split_path = Path(cfg["data"]["split_file"]) if cfg["data"].get("split_file") and Path(cfg["data"]["split_file"]).is_file() else None
    train_cases, val_cases, test_cases = discover_and_split(data_root, split_file=split_path)

    if len(train_cases) == 0:
        raise ValueError(f"No valid patient cases discovered in '{data_root}'. Verify dataset structure.")

    print(f"Discovered patient split: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}")

    modalities = list(MVP_MODALITIES)
    patch_size = tuple(cfg["data"]["patch_size"])
    samples_per_vol = cfg["data"]["samples_per_volume"]

    train_dicts = cases_to_monai_dicts(train_cases, modalities)
    val_dicts = cases_to_monai_dicts(val_cases, modalities)

    train_ds = Dataset(data=train_dicts, transform=build_gpu_train_transforms(patch_size, samples_per_vol))
    val_ds = Dataset(data=val_dicts, transform=build_gpu_val_transforms(patch_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=pad_list_data_collate,
    )

    # 4. Model construction with architecture metadata
    arch_config = {
        "model_name": cfg["model"]["name"],
        "spatial_dims": 3,
        "in_channels": cfg["model"]["in_channels"],
        "out_channels": cfg["model"]["out_channels"],
        "channels": tuple(cfg["model"]["channels"]),
        "strides": tuple(cfg["model"]["strides"]),
        "num_res_units": cfg["model"]["num_res_units"],
        "dropout": cfg["model"]["dropout"],
    }

    model = Unet3D(
        in_channels=arch_config["in_channels"],
        out_channels=arch_config["out_channels"],
        channels=arch_config["channels"],
        strides=arch_config["strides"],
        num_res_units=arch_config["num_res_units"],
        dropout=arch_config["dropout"],
    ).to(device)

    loss_fn = ClassWeightedDiceCELoss(
        dice_weight=cfg["loss"]["dice_weight"],
        ce_weight=cfg["loss"]["ce_weight"],
        include_background=cfg["loss"]["include_background"],
        dice_class_weights=tuple(cfg["loss"]["dice_class_weights"]),
        ce_class_weights=tuple(cfg["loss"]["ce_class_weights"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("min_lr", 1e-6),
    )
    use_amp = bool(cfg["training"]["mixed_precision"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")
    no_improve = 0

    # 5. Handle optional resume
    resume_path = cfg["training"].get("resume")
    if resume_path:
        r_path = Path(resume_path)
        if not r_path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume '{r_path}' not found.")
        print(f"Resuming training from checkpoint: {r_path}")
        checkpoint = torch.load(r_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
        print(f"Successfully resumed at epoch {start_epoch} (best_val_loss: {best_val_loss:.4f})")

    epochs = cfg["training"]["epochs"]
    grad_accum = cfg["training"]["gradient_accumulation_steps"]
    patience = cfg["training"]["patience"]

    print(f"Starting GPU training from epoch {start_epoch} to {epochs} with patch size {patch_size}...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            if hasattr(x, "as_tensor"):
                x = x.as_tensor()
            if hasattr(y, "as_tensor"):
                y = y.as_tensor()
            if y.ndim == 5 and y.shape[1] == 1:
                y = y.squeeze(1)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += float(loss.item() * grad_accum)
            batches += 1

        avg_train = train_loss / max(1, batches)

        # Validation on patch crops
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                if hasattr(x, "as_tensor"):
                    x = x.as_tensor()
                if hasattr(y, "as_tensor"):
                    y = y.as_tensor()
                if y.ndim == 5 and y.shape[1] == 1:
                    y = y.squeeze(1)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                val_loss += float(loss.item())
                val_batches += 1

        avg_val = val_loss / max(1, val_batches)
        scheduler.step()

        log_msg = f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"

        # Optional periodic sliding-window whole-volume validation
        if cfg["training"].get("val_sliding_window") and (epoch % cfg["training"].get("val_interval_epochs", 5) == 0):
            sw_metrics = evaluate_sliding_window_validation(
                model=model,
                val_cases=val_cases,
                modalities=modalities,
                roi_size=patch_size,
                device=device,
                use_amp=use_amp,
            )
            log_msg += f" | Val WT Dice: {sw_metrics['val_wt_dice']:.4f} | TC Dice: {sw_metrics['val_tc_dice']:.4f} | ET Dice: {sw_metrics['val_et_dice']:.4f} [Monitoring metric]"

        print(log_msg)

        # Construct full checkpoint dictionary preserving architecture and optimizer states
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "val_loss": avg_val,
            "best_val_loss": best_val_loss,
            "model_name": "unet_3d",
            "arch_config": arch_config,
            "channels": arch_config["channels"],
            "patch_size": patch_size,
            "seed": cfg["training"]["seed"],
        }

        # Save last.pt atomically at every epoch
        atomic_save(checkpoint_dict, save_dir / "last.pt")

        # Save best.pt atomically when validation loss improves
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            checkpoint_dict["best_val_loss"] = best_val_loss
            atomic_save(checkpoint_dict, save_dir / "best.pt")
            no_improve = 0
            print(f"  -> Best model checkpoint saved to {save_dir / 'best.pt'} (val_loss: {avg_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs).")
                break

    print(f"GPU Training complete. Best Val Loss: {best_val_loss:.4f}. Outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()

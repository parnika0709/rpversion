
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from braintumor.config import config
from braintumor.data.splits import discover_and_split
from braintumor.data.dataset import BratsDataset
from braintumor.data.slicing import SlicingDataset
from braintumor.data.preprocessing import train_transforms, val_transforms
from braintumor.models.attention_unet_2d import AttentionUnet2D, MVP_MODALITIES
from braintumor.training.train import TrainConfig, fit


def main():
    device = config.device
    print("Device:", device)

    # Use only a small subset for this controlled test.
    train_cases, val_cases, _ = discover_and_split(
        config.data_root,
        split_file=config.split_file,
    )

    train_cases = train_cases[:2]
    val_cases = val_cases[:1]

    print("Training patients:", len(train_cases))
    print("Validation patients:", len(val_cases))

    if not train_cases or not val_cases:
        raise RuntimeError("Insufficient patients for the test.")

    modalities = list(MVP_MODALITIES)

    train_base = BratsDataset(
        train_cases,
        modalities,
        transform=train_transforms(modalities),
    )

    val_base = BratsDataset(
        val_cases,
        modalities,
        transform=val_transforms(modalities),
    )

    train_dataset = SlicingDataset(
        train_base,
        training=True,
        samples_per_volume=2,
        foreground_probability=0.7,
    )

    val_dataset = SlicingDataset(
        val_base,
        training=False,
        slices_per_volume=2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    print("Training slices:", len(train_dataset))
    print("Validation slices:", len(val_dataset))

    model = AttentionUnet2D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.model_channels,
        strides=config.model_strides,
        dropout=config.dropout,
    )

    test_config = TrainConfig(
        epochs=1,
        batch_size=2,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed,
        log_interval=1,
        save_dir=Path("runs/one_epoch_test"),
        num_workers=0,
    )

    history = fit(
        model,
        train_loader,
        val_loader,
        test_config,
        device=device,
    )

    print("\nONE EPOCH TEST RESULTS")
    print("Training loss:", history["train_loss"])
    print("Validation loss:", history["val_loss"])

    checkpoint_dir = Path("runs/one_epoch_test")

    print("Last checkpoint exists:", (checkpoint_dir / "last.pt").exists())
    print("Best checkpoint exists:", (checkpoint_dir / "best.pt").exists())

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert torch.isfinite(torch.tensor(history["train_loss"][0]))
    assert torch.isfinite(torch.tensor(history["val_loss"][0]))
    assert (checkpoint_dir / "last.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()

    print("\nONE EPOCH REAL-DATA TEST PASSED")


if __name__ == "__main__":
    main()

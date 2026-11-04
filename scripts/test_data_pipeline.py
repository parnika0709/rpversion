
from torch.utils.data import DataLoader

from braintumor.config import config
from braintumor.data.splits import discover_and_split
from braintumor.data.dataset import BratsDataset
from braintumor.data.slicing import SlicingDataset
from braintumor.data.preprocessing import train_transforms, val_transforms
from braintumor.models.attention_unet_2d import MVP_MODALITIES


def main():
    print("Loading real BraTS dataset...")

    train_cases, val_cases, test_cases = discover_and_split(
        config.data_root,
        split_file=config.split_file,
    )

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
        samples_per_volume=8,
        foreground_probability=0.7,
    )

    val_dataset = SlicingDataset(
        val_base,
        training=False,
        slices_per_volume=8,
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

    print("\nDataset sizes:")
    print("Training patients:", len(train_cases))
    print("Validation patients:", len(val_cases))
    print("Test patients:", len(test_cases))
    print("Training slices:", len(train_dataset))
    print("Validation slices:", len(val_dataset))

    print("\nChecking training batch...")
    train_batch = next(iter(train_loader))

    print("Training image shape:", tuple(train_batch["image"].shape))
    print("Training label shape:", tuple(train_batch["label"].shape))
    print("Training image dtype:", train_batch["image"].dtype)
    print("Training label dtype:", train_batch["label"].dtype)
    print("Training image finite:", train_batch["image"].isfinite().all().item())
    print("Training labels:", train_batch["label"].unique().tolist())

    print("\nChecking validation batch...")
    val_batch = next(iter(val_loader))

    print("Validation image shape:", tuple(val_batch["image"].shape))
    print("Validation label shape:", tuple(val_batch["label"].shape))
    print("Validation image finite:", val_batch["image"].isfinite().all().item())
    print("Validation labels:", val_batch["label"].unique().tolist())

    assert train_batch["image"].shape[1] == config.in_channels
    assert train_batch["label"].ndim == 3
    assert train_batch["image"].shape[0] == train_batch["label"].shape[0]
    assert train_batch["image"].shape[2:] == train_batch["label"].shape[1:]
    assert train_batch["image"].isfinite().all()
    assert train_batch["label"].unique().max().item() <= 3
    assert train_batch["label"].unique().min().item() >= 0

    assert val_batch["image"].shape[1] == config.in_channels
    assert val_batch["image"].shape[0] == val_batch["label"].shape[0]
    assert val_batch["image"].shape[2:] == val_batch["label"].shape[1:]
    assert val_batch["image"].isfinite().all()
    assert val_batch["label"].unique().max().item() <= 3
    assert val_batch["label"].unique().min().item() >= 0

    print("\nDATA PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()

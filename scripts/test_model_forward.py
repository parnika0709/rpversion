
import torch

from braintumor.config import config
from braintumor.data.splits import discover_and_split
from braintumor.data.dataset import BratsDataset
from braintumor.data.slicing import SlicingDataset
from braintumor.data.preprocessing import train_transforms
from braintumor.models.attention_unet_2d import (
    AttentionUnet2D,
    MVP_MODALITIES,
)
from braintumor.training.losses import DiceCELoss
from torch.utils.data import DataLoader


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Testing device:", device)

    train_cases, _, _ = discover_and_split(
        config.data_root,
        split_file=config.split_file,
    )

    modalities = list(MVP_MODALITIES)

    base_dataset = BratsDataset(
        train_cases[:1],
        modalities,
        transform=train_transforms(modalities),
    )

    slice_dataset = SlicingDataset(
        base_dataset,
        training=True,
        samples_per_volume=2,
    )

    loader = DataLoader(
        slice_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    model = AttentionUnet2D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.model_channels,
        strides=config.model_strides,
        dropout=config.dropout,
    ).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(images)
        loss = DiceCELoss()(logits, labels)

    print("\nMODEL FORWARD TEST")
    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))
    print("Target shape:", tuple(labels.shape))
    print("Loss:", loss.item())
    print("Loss finite:", torch.isfinite(loss).item())
    print("Output finite:", torch.isfinite(logits).all().item())

    assert logits.shape == (
        images.shape[0],
        config.out_channels,
        images.shape[2],
        images.shape[3],
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    print("\nMODEL FORWARD TEST PASSED")


if __name__ == "__main__":
    main()


import torch
import tempfile
from pathlib import Path

from braintumor.config import config
from braintumor.models.attention_unet_2d import AttentionUnet2D
from braintumor.training.train import (
    TrainConfig,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
)
from braintumor.training.losses import DiceCELoss


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Testing device:", device)

    model = AttentionUnet2D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.model_channels,
        strides=config.model_strides,
        dropout=config.dropout,
    ).to(device)

    train_config = TrainConfig(
        epochs=2,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=1e-5,
        save_dir=Path("runs/test_training_setup"),
    )

    optimizer = build_optimizer(model, train_config)

    scheduler = build_scheduler(
        optimizer,
        cfg=train_config,
        steps_per_epoch=2,
    )

    images = torch.randn(2, 3, 240, 240).to(device)
    labels = torch.randint(0, 4, (2, 240, 240)).to(device)

    criterion = DiceCELoss()

    model.train()
    optimizer.zero_grad()

    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()
    scheduler.step()

    print("\nTRAINING SETUP TEST")
    print("Optimizer:", type(optimizer).__name__)
    print("Scheduler:", type(scheduler).__name__)
    print("Loss:", loss.item())
    print("Loss finite:", torch.isfinite(loss).item())

    assert torch.isfinite(loss).item()

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"

        save_checkpoint(
            model=model,
            cfg=train_config,
            epoch=1,
            val_loss=loss.item(),
            path=checkpoint_path,
        )

        print("Checkpoint exists:", checkpoint_path.exists())
        assert checkpoint_path.exists()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        assert "state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_loss" in checkpoint

        print("Checkpoint loading: PASSED")
        print("Saved epoch:", checkpoint["epoch"])

    print("\nTRAINING SETUP TEST PASSED")


if __name__ == "__main__":
    main()

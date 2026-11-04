"""Loss functions for BraTS segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Combined Dice + weighted Cross-Entropy loss."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-5,
        class_weights: tuple[float, ...] | None = (
            0.10,
            1.68,
            0.81,
            1.42,
        ),
    ) -> None:
        super().__init__()

        if dice_weight < 0 or ce_weight < 0:
            raise ValueError("Loss weights must be non-negative.")

        if dice_weight + ce_weight <= 0:
            raise ValueError("At least one loss component must be enabled.")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.smooth = smooth

        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)

            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Class weights must be finite and positive.")

            self.register_buffer("class_weights", weights)
        else:
            self.register_buffer("class_weights", None)

    def dice_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate per-class soft Dice loss."""

        probs = torch.softmax(logits, dim=1)
        n_classes = logits.shape[1]

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=n_classes,
        ).float()

        dims = list(range(target_one_hot.ndim))
        channel_dim = dims.pop()
        dims.insert(1, channel_dim)

        target_one_hot = target_one_hot.permute(*dims)

        reduction_dims = [0] + list(range(2, target_one_hot.ndim))

        intersection = (
            probs * target_one_hot
        ).sum(dim=reduction_dims)

        cardinality = (
            probs + target_one_hot
        ).sum(dim=reduction_dims)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            cardinality + self.smooth
        )

        if not self.include_background:
            dice = dice[1:]

        return 1.0 - dice.mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined weighted CE and Dice loss."""

        if logits.ndim < 4:
            raise ValueError(
                f"Expected logits with shape (B,C,...), got {tuple(logits.shape)}"
            )

        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"Target shape {tuple(target.shape)} does not match "
                f"logits spatial shape {tuple(logits.shape[:1] + logits.shape[2:])}"
            )

        if self.class_weights is not None:
            if logits.shape[1] != self.class_weights.numel():
                raise ValueError(
                    f"Expected {self.class_weights.numel()} output classes, "
                    f"got {logits.shape[1]}"
                )

        ce = F.cross_entropy(
            logits,
            target.long(),
            weight=self.class_weights,
        )

        dl = self.dice_loss(logits, target)

        return self.dice_weight * dl + self.ce_weight * ce

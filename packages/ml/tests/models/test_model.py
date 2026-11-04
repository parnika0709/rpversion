import torch
import pytest
from braintumor.models.attention_unet_2d import AttentionUnet2D

def test_model_forward_pass():
    """Test that AttentionUnet2D produces correct output shapes."""
    batch_size = 2
    in_channels = 3
    h, w = 128, 128
    out_channels = 4

    model = AttentionUnet2D(in_channels=in_channels, out_channels=out_channels)
    model.eval()

    x = torch.randn(batch_size, in_channels, h, w)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, out_channels, h, w)

def test_model_invalid_config():
    """Test that AttentionUnet2D raises error for invalid channels/strides."""
    with pytest.raises(ValueError):
        # channels len must be strides len + 1
        AttentionUnet2D(channels=(32, 64), strides=(2, 2, 2))

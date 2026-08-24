import pytest
import torch
from torch import nn

from regnav.models.vjepa_adapter import VJEPA2Adapter


class FakeVideoBackbone(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.input_shape = None

    def forward(self, video):
        self.input_shape = tuple(video.shape)
        pooled = video.mean(dim=(2, 3, 4))
        return self.projection(pooled)[:, None].repeat(1, 6, 1)


def test_adapter_returns_video_patch_tokens():
    backbone = FakeVideoBackbone()
    adapter = VJEPA2Adapter(
        "vjepa2_1_fake",
        image_size=(32, 32),
        patch_size=16,
        context_frames=4,
        backbone=backbone,
    )
    video = torch.randn(2, 3, 4, 32, 32)

    output = adapter(video)

    assert backbone.input_shape == (2, 3, 4, 32, 32)
    assert output.shape == (2, 6, 4)


def test_adapter_rejects_partial_tubelet():
    with pytest.raises(ValueError, match="tubelet"):
        VJEPA2Adapter(
            "vjepa2_1_fake",
            image_size=(32, 32),
            patch_size=16,
            context_frames=3,
            backbone=FakeVideoBackbone(),
        )

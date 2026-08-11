import pytest
import torch
from torch import nn

from regnav.models.vit_adapter import DinoV2Adapter


class FakeTokenBackbone(nn.Module):
    num_prefix_tokens = 5
    embed_dim = 4

    def forward_features(self, image):
        tokens = torch.arange(44, dtype=image.dtype, device=image.device).view(1, 11, 4)
        return tokens.repeat(image.shape[0], 1, 1)


def test_adapter_removes_cls_and_register_tokens():
    adapter = DinoV2Adapter(
        model_name="fake", image_size=(28, 42), patch_size=14, backbone=FakeTokenBackbone()
    )

    output = adapter(torch.randn(2, 3, 42, 28))

    assert output.shape == (2, 6, 4)
    torch.testing.assert_close(output[0, 0], torch.tensor([20.0, 21.0, 22.0, 23.0]))


def test_adapter_rejects_image_size_not_divisible_by_patch():
    with pytest.raises(ValueError, match="divisible"):
        DinoV2Adapter(
            model_name="fake",
            image_size=(28, 40),
            patch_size=14,
            backbone=FakeTokenBackbone(),
        )

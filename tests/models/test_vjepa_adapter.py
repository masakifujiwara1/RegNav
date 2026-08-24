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

def test_adapter_loads_official_checkpoint_without_broken_hub_url(monkeypatch):
    hub_call = {}
    checkpoint_call = {}
    backbone = FakeVideoBackbone()
    checkpoint = {
        "ema_encoder": {
            f"module.backbone.{name}": torch.ones_like(value)
            for name, value in backbone.state_dict().items()
        }
    }

    def fake_load(repository, model_name, **kwargs):
        hub_call.update(repository=repository, model_name=model_name, **kwargs)
        return backbone, nn.Identity()

    def fake_checkpoint(url, **kwargs):
        checkpoint_call.update(url=url, **kwargs)
        return checkpoint

    monkeypatch.setattr(torch.hub, "load", fake_load)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_checkpoint)

    VJEPA2Adapter(
        "vjepa2_1_vit_base_384",
        image_size=(384, 384),
        patch_size=16,
        context_frames=4,
    )

    assert hub_call["repository"].endswith(":45d025f")
    assert hub_call["pretrained"] is False
    assert hub_call["skip_validation"] is True
    assert checkpoint_call["url"].startswith("https://dl.fbaipublicfiles.com/")
    assert all(torch.all(parameter == 1) for parameter in backbone.parameters())

from torch import nn

from regnav.models.vit_adapter import DinoV2Adapter


class FakeBackbone(nn.Module):
    pass


def test_production_backbone_uses_configured_image_size(monkeypatch):
    captured = {}

    def create_model(name, **kwargs):
        captured.update(kwargs)
        return FakeBackbone()

    monkeypatch.setattr("timm.create_model", create_model)

    DinoV2Adapter("fake", image_size=(28, 42), patch_size=14)

    assert captured["img_size"] == (42, 28)

from torch import nn

from regnav.config import ModelConfig
from regnav.models.factory import build_model
from regnav.models.regnav import RegNav


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()


class FakeDinoBackbone(nn.Module):
    num_prefix_tokens = 5
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock()])


def test_factory_builds_regnav_with_injected_backbone():
    config = ModelConfig(
        image_size=(28, 42),
        backbone="fake_vit",
        backbone_dim=4,
        patch_size=14,
        d_model=16,
        d_ffn=32,
        num_heads=4,
        lora_rank=2,
    )

    model = build_model(config, backbone=FakeDinoBackbone())

    assert isinstance(model, RegNav)

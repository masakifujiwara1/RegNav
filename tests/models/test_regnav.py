import torch
from torch import nn

from regnav.config import ModelConfig
from regnav.models.lora import LoRALinear
from regnav.models.regnav import RegNav


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)

    def forward(self, tokens):
        query, _, _ = self.qkv(tokens).chunk(3, dim=-1)
        return self.proj(query)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()

    def forward(self, tokens):
        return self.attn(tokens)


class FakeDinoBackbone(nn.Module):
    num_prefix_tokens = 5
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(3, 4)
        self.blocks = nn.ModuleList([FakeBlock()])

    def forward_features(self, image):
        pooled = image.mean(dim=(-2, -1))
        tokens = self.patch(pooled)[:, None].repeat(1, 11, 1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class FakeVJEPA2Backbone(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(3, 4)
        self.blocks = nn.ModuleList([FakeBlock()])
        self.input_shape = None

    def forward(self, video):
        self.input_shape = tuple(video.shape)
        pooled = video.mean(dim=(-3, -2, -1))
        tokens = self.patch(pooled)[:, None].repeat(1, 6, 1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


def _config():
    return ModelConfig(
        image_size=(28, 42),
        backbone="fake_vit",
        backbone_dim=4,
        patch_size=14,
        d_model=16,
        d_ffn=32,
        num_heads=4,
        lora_rank=2,
    )


def _batch():
    return {
        "image": torch.randn(2, 3, 42, 28),
        "ego": torch.randn(2, 4),
        "route_goal": torch.randn(2, 6),
        "target_trajectory": torch.zeros(2, 8, 3),
        "valid_mask": torch.ones(2, 8, dtype=torch.bool),
    }


def test_regnav_output_selection_and_trainability():
    model = RegNav(_config(), backbone=FakeDinoBackbone())

    output = model(_batch())
    output["trajectory"].sum().backward()
    selected = output["proposals"][
        torch.arange(2), output["scores"].argmax(dim=-1)
    ]

    assert output["trajectory"].shape == (2, 8, 3)
    assert output["proposals"].shape == (2, 8, 8, 3)
    assert output["scores"].shape == (2, 8)
    assert output["score_components"].shape == (2, 8, 4)
    assert len(output["refinements"]) == 2
    assert torch.isfinite(output["trajectory"]).all()
    torch.testing.assert_close(output["trajectory"], selected)
    assert model.scene_pool.queries.grad is not None
    assert any(
        module.lora_b.weight.grad is not None
        for module in model.modules()
        if isinstance(module, LoRALinear)
    )
    assert model.backbone.blocks[0].attn.qkv.base.weight.grad is None


def test_regnav_selects_vjepa_adapter_for_video_backbone():
    config = ModelConfig(
        image_size=(28, 42),
        backbone="vjepa2_1_fake",
        backbone_dim=4,
        patch_size=14,
        context_frames=4,
        d_model=16,
        d_ffn=32,
        num_heads=4,
        lora_rank=2,
    )
    backbone = FakeVJEPA2Backbone()
    batch = _batch()
    batch["image"] = torch.randn(2, 3, 4, 42, 28)

    output = RegNav(config, backbone=backbone)(batch)

    assert backbone.input_shape == (2, 3, 4, 42, 28)
    assert output["trajectory"].shape == (2, 8, 3)


def test_regnav_eval_omits_refinements():
    model = RegNav(_config(), backbone=FakeDinoBackbone()).eval()

    with torch.no_grad():
        output = model(_batch())

    assert "refinements" not in output


def test_regnav_cached_scene_inference_matches_forward():
    model = RegNav(_config(), backbone=FakeDinoBackbone()).eval()
    batch = _batch()

    with torch.no_grad():
        expected = model(batch)
        scene = model.encode_scene(batch["image"])
        actual = model.decode_scene(scene, batch["ego"], batch["route_goal"])

    assert scene.shape == (2, model.config.num_scene_registers, model.config.d_model)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name])

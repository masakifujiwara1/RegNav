import torch
from torch import nn

from regnav.config import ModelConfig
from regnav.models.lite import RegNavLite


class FakeMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 1)

    def forward(self, image):
        return self.conv(image)


class ConstantDeltaHead(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))

    def forward(self, features):
        return self.delta.repeat(features.shape[0], 8)


def _config():
    return ModelConfig(
        name="regnav_lite",
        image_size=(32, 18),
        backbone="mobilenet_v3_small",
        backbone_dim=16,
        patch_size=1,
        num_proposals=1,
        num_scene_registers=0,
        d_model=128,
        d_ffn=256,
        decoder_layers=1,
        num_refinements=1,
        scorer_layers=1,
        lora_rank=0,
    )


def _batch(batch_size=2):
    return {
        "image": torch.randn(batch_size, 3, 18, 32),
        "ego": torch.randn(batch_size, 4),
        "route_goal": torch.randn(batch_size, 6),
        "target_trajectory": torch.zeros(batch_size, 8, 3),
        "valid_mask": torch.ones(batch_size, 8, dtype=torch.bool),
    }


def test_lite_output_contract_and_gradient():
    model = RegNavLite(_config(), backbone=FakeMobileNet())

    output = model(_batch())
    output["trajectory"].sum().backward()

    assert output["trajectory"].shape == (2, 8, 3)
    assert output["proposals"].shape == (2, 1, 8, 3)
    assert output["scores"].shape == (2, 1)
    assert output["score_components"].shape == (2, 1, 4)
    assert torch.isfinite(output["trajectory"]).all()
    assert model.backbone.conv.weight.grad is not None


def test_lite_integrates_relative_pose_deltas():
    model = RegNavLite(_config(), backbone=FakeMobileNet())
    model.trajectory_head = ConstantDeltaHead([0.1, 0.0, 0.0])

    output = model(_batch(1))["trajectory"]

    torch.testing.assert_close(output[0, :, 0], torch.arange(1, 9) * 0.1)

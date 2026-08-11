import torch

from regnav.config import ModelConfig
from regnav.models.registers import SceneRegisterPool, TrajectoryRegisterDecoder


def small_config():
    return ModelConfig(
        image_size=(28, 42),
        backbone="fake_vit",
        backbone_dim=4,
        patch_size=14,
        d_model=16,
        d_ffn=32,
        num_heads=4,
    )


def test_scene_register_pool_shape():
    pool = SceneRegisterPool(num_registers=8, input_dim=4, d_model=16, num_heads=4)

    output = pool(torch.randn(2, 6, 4))

    assert output.shape == (2, 8, 16)


def test_decoder_returns_both_refinements():
    decoder = TrajectoryRegisterDecoder(small_config())

    proposals = decoder(torch.randn(2, 8, 16), torch.randn(2, 8, 16))

    assert len(proposals) == 2
    assert proposals[-1].shape == (2, 8, 8, 3)
    assert torch.isfinite(proposals[-1]).all()

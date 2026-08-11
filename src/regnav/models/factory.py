from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from regnav.config import ModelConfig
from regnav.models.lite import RegNavLite
from regnav.models.regnav import RegNav


def build_model(config: ModelConfig, backbone: nn.Module | None = None) -> nn.Module:
    if config.name == "regnav_lite":
        if backbone is None:
            backbone = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
            ).features
        return RegNavLite(config, backbone)
    if config.name == "regnav":
        return RegNav(config, backbone)
    raise ValueError(f"unknown model: {config.name}")

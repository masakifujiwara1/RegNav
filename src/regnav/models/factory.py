from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from regnav.config import ModelConfig
from regnav.models.lite import RegNavLite


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "regnav_lite":
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        ).features
        return RegNavLite(config, backbone)
    raise ValueError(f"unknown model: {config.name}")

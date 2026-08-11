from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def _yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


@dataclass(frozen=True)
class ModelConfig:
    name: str = "regnav"
    image_size: tuple[int, int] = (448, 252)
    backbone: str = "vit_small_patch14_reg4_dinov2"
    backbone_dim: int = 384
    patch_size: int = 14
    num_poses: int = 8
    interval: float = 0.5
    horizon: float = 4.0
    ego_features: int = 4
    route_features: int = 6
    num_proposals: int = 8
    num_scene_registers: int = 8
    d_model: int = 256
    d_ffn: int = 1024
    num_heads: int = 8
    decoder_layers: int = 4
    num_refinements: int = 2
    scorer_layers: int = 2
    lora_rank: int = 8
    scorer_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        positive = (
            "backbone_dim",
            "patch_size",
            "num_poses",
            "ego_features",
            "route_features",
            "num_proposals",
            "d_model",
            "d_ffn",
            "num_heads",
            "decoder_layers",
            "num_refinements",
            "scorer_layers",
        )
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("model dimensions must be positive")
        if self.interval <= 0 or self.horizon <= 0:
            raise ValueError("interval and horizon must be positive")
        if self.num_poses * self.interval != self.horizon:
            raise ValueError("num_poses * interval must equal horizon")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("image_size must contain two positive dimensions")
        if self.backbone.startswith("vit_") and any(
            value % self.patch_size for value in self.image_size
        ):
            raise ValueError("image_size must be divisible by patch_size")
        if self.ego_features != 4 or self.route_features != 6:
            raise ValueError("expected four ego and six route features")
        if len(self.scorer_weights) != 4:
            raise ValueError("scorer_weights must contain four values")
        if self.lora_rank < 0 or self.num_scene_registers < 0:
            raise ValueError("adapter dimensions cannot be negative")

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        data = _yaml(path)
        if "image_size" in data:
            data["image_size"] = tuple(data["image_size"])
        if "scorer_weights" in data:
            data["scorer_weights"] = tuple(data["scorer_weights"])
        return cls(**data)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 3e-4
    lora_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    lora_start_epoch: int = 2
    gradient_clip_norm: float = 1.0
    workers: int = 4
    trajectory_weight: float = 1.0
    heading_weight: float = 0.5
    smoothness_weight: float = 0.1
    diversity_weight: float = 0.1
    refinement_weight: float = 0.5
    scorer_weight: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            "epochs",
            "batch_size",
            "learning_rate",
            "lora_learning_rate",
            "gradient_clip_norm",
        )
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("training values must be positive")
        if self.seed < 0 or self.workers < 0 or self.lora_start_epoch < 0:
            raise ValueError("seed, workers, and lora_start_epoch cannot be negative")
        weights = [
            getattr(self, field.name)
            for field in fields(self)
            if field.name.endswith("_weight")
        ]
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights cannot be negative")

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        return cls(**_yaml(path))


def load_yaml_config(
    model_path: Path, training_path: Path
) -> tuple[ModelConfig, TrainingConfig]:
    return ModelConfig.from_yaml(model_path), TrainingConfig.from_yaml(training_path)

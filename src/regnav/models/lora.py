import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.enabled = True
        self.merged = False
        self.base.requires_grad_(False)

    @classmethod
    def from_linear(
        cls, layer: nn.Linear, rank: int, alpha: float | None = None
    ) -> "LoRALinear":
        return cls(layer, rank, alpha)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if self.enabled and not self.merged:
            output = output + self.scale * self.lora_b(self.lora_a(inputs))
        return output

    def merge(self) -> None:
        if self.merged:
            return
        with torch.no_grad():
            self.base.weight.add_(
                self.scale * self.lora_b.weight @ self.lora_a.weight
            )
        self.merged = True


def inject_attention_lora(module: nn.Module, rank: int) -> None:
    module.requires_grad_(False)
    attention_modules = [
        child
        for child in module.modules()
        if isinstance(getattr(child, "qkv", None), nn.Linear)
        and isinstance(getattr(child, "proj", None), nn.Linear)
    ]
    for attention in attention_modules:
        attention.qkv = LoRALinear.from_linear(attention.qkv, rank)
        attention.proj = LoRALinear.from_linear(attention.proj, rank)


def set_lora_enabled(module: nn.Module, enabled: bool) -> None:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.enabled = enabled
            child.lora_a.requires_grad_(enabled and not child.merged)
            child.lora_b.requires_grad_(enabled and not child.merged)


def merge_lora(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.merge()

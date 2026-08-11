import torch
from torch import nn

from regnav.models.lora import LoRALinear, inject_attention_lora, merge_lora


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()


class ToyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block()])


def test_lora_freezes_base_and_trains_only_adapters():
    model = ToyViT()

    inject_attention_lora(model, rank=2)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "blocks.0.attn.qkv.lora_a.weight",
        "blocks.0.attn.qkv.lora_b.weight",
        "blocks.0.attn.proj.lora_a.weight",
        "blocks.0.attn.proj.lora_b.weight",
    }


def test_merge_preserves_output_and_is_idempotent():
    layer = LoRALinear.from_linear(nn.Linear(4, 3), rank=2)
    nn.init.normal_(layer.lora_b.weight)
    x = torch.randn(2, 4)
    before = layer(x)

    merge_lora(layer)
    once = layer(x)
    merge_lora(layer)

    torch.testing.assert_close(before, once)
    torch.testing.assert_close(once, layer(x))

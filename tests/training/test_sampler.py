from collections import Counter

from torch import nn

from regnav.models.lora import LoRALinear
from regnav.training.engine import set_training_stage
from regnav.training.sampler import DomainBalancedSampler


def test_balanced_sampler_equalizes_datasets():
    labels = ["large"] * 100 + ["small"] * 10
    sampler = DomainBalancedSampler(labels=labels, num_samples=200, seed=3)

    sampled = Counter(labels[index] for index in sampler)

    assert abs(sampled["large"] - sampled["small"]) <= 20


def test_lora_starts_after_warmup():
    model = nn.Sequential(LoRALinear.from_linear(nn.Linear(4, 4), rank=2))

    set_training_stage(model, epoch=0, lora_start_epoch=2)
    assert not model[0].lora_a.weight.requires_grad
    assert not model[0].lora_b.weight.requires_grad

    set_training_stage(model, epoch=2, lora_start_epoch=2)
    assert model[0].lora_a.weight.requires_grad
    assert model[0].lora_b.weight.requires_grad

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from regnav.config import ModelConfig, TrainingConfig
from regnav.models.lite import RegNavLite
from regnav.training.engine import (
    _autocast_dtype,
    _to_device,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
    validation_ade,
)


class FakeMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, image):
        return self.conv(image)


class FixedTrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return {
            "trajectory": torch.zeros_like(batch["target_trajectory"]) + self.anchor
        }


def _model_config():
    return ModelConfig(
        name="regnav_lite",
        image_size=(16, 12),
        backbone="mobilenet_v3_small",
        backbone_dim=8,
        patch_size=1,
        num_proposals=1,
        num_scene_registers=0,
        d_model=32,
        d_ffn=64,
        decoder_layers=1,
        num_refinements=1,
        scorer_layers=1,
        lora_rank=0,
    )


def _batch():
    return {
        "image": torch.randn(2, 3, 12, 16),
        "ego": torch.randn(2, 4),
        "route_goal": torch.randn(2, 6),
        "target_trajectory": torch.randn(2, 8, 3),
        "valid_mask": torch.ones(2, 8, dtype=torch.bool),
    }


def test_to_device_requests_non_blocking_cuda_transfer():
    class TensorSpy:
        def to(self, device, *, non_blocking):
            self.call = (device, non_blocking)
            return self

    tensor = TensorSpy()

    assert _to_device({"image": tensor}, torch.device("cuda"))["image"] is tensor
    assert tensor.call == (torch.device("cuda"), True)


def test_autocast_dtype_prefers_bfloat16_when_cuda_supports_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert _autocast_dtype(torch.device("cuda")) is torch.bfloat16


def test_autocast_dtype_falls_back_to_float16_without_bfloat16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert _autocast_dtype(torch.device("cuda")) is torch.float16


def test_validation_ade_aggregates_real_trajectory_errors():
    batch = _batch()
    batch["target_trajectory"] = torch.zeros(2, 8, 3)
    batch["target_trajectory"][..., 0] = 2.0

    result = validation_ade(
        FixedTrajectoryModel(), [batch, batch], torch.device("cpu")
    )

    assert result == pytest.approx(2.0)


def test_one_epoch_checkpoint_round_trip(tmp_path):
    torch.manual_seed(3)
    model_config = _model_config()
    training_config = replace(TrainingConfig(), batch_size=2, workers=0)
    model = RegNavLite(model_config, FakeMobileNet())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader([_batch() for _ in range(4)], batch_size=None)

    losses = train_epoch(model, loader, optimizer, training_config, torch.device("cpu"))
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        model_config=model_config,
        training_config=training_config,
        manifest_hash="abc",
        best_metric_name="validation/worst_ade",
        best_metric=0.25,
    )
    inference_batch = _batch()
    expected = model(inference_batch)["trajectory"]
    restored = RegNavLite(model_config, FakeMobileNet())
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    metadata = load_checkpoint(
        checkpoint,
        restored,
        restored_optimizer,
        expected_model_config=model_config,
        expected_manifest_hash="abc",
    )

    assert torch.isfinite(torch.tensor(losses["total"]))
    assert checkpoint.exists()
    assert metadata["epoch"] == 1
    assert metadata["best_metric_name"] == "validation/worst_ade"
    assert metadata["best_metric"] == 0.25
    torch.testing.assert_close(expected, restored(inference_batch)["trajectory"])


def test_checkpoint_rejects_manifest_mismatch(tmp_path):
    config = _model_config()
    model = RegNavLite(config, FakeMobileNet())
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        None,
        epoch=0,
        model_config=config,
        training_config=TrainingConfig(),
        manifest_hash="old",
    )

    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint(
            checkpoint,
            model,
            expected_model_config=config,
            expected_manifest_hash="new",
        )

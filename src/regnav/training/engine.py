from dataclasses import asdict
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from regnav.config import ModelConfig, TrainingConfig
from regnav.contracts import RegNavBatch
from regnav.metrics import ade
from regnav.models.lora import set_lora_enabled
from regnav.training.losses import compute_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_training_stage(model: nn.Module, epoch: int, lora_start_epoch: int) -> None:
    set_lora_enabled(model, epoch >= lora_start_epoch)


def _autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _to_device(batch: RegNavBatch, device: torch.device) -> RegNavBatch:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }  # type: ignore[return-value]


def train_epoch(
    model: nn.Module,
    loader: Iterable[RegNavBatch],
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    model.to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device.type, dtype=_autocast_dtype(device), enabled=device.type == "cuda"
        ):
            losses = compute_loss(model(batch), batch, config)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        batches += 1
    if not batches:
        raise ValueError("loader must contain at least one batch")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def validation_ade(
    model: nn.Module,
    loader: Iterable[RegNavBatch],
    device: torch.device,
) -> float:
    model.eval()
    model.to(device)
    predictions = []
    targets = []
    masks = []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.autocast(
            device.type, dtype=_autocast_dtype(device), enabled=device.type == "cuda"
        ):
            output = model(batch)
        predictions.append(output["trajectory"].cpu())
        targets.append(batch["target_trajectory"].cpu())
        masks.append(batch["valid_mask"].cpu())
    if not predictions:
        raise ValueError("loader must contain at least one batch")
    return float(
        ade(
            torch.cat(predictions),
            torch.cat(targets),
            torch.cat(masks),
        ).item()
    )


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: Iterable[RegNavBatch],
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    model.to(device)
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = _to_device(batch, device)
        losses = compute_loss(model(batch), batch, config)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
    if not batches:
        raise ValueError("loader must contain at least one batch")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    manifest_hash: str,
    git_revision: str | None = None,
    best_metric_name: str | None = None,
    best_metric: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "manifest_hash": manifest_hash,
            "git_revision": git_revision,
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_model_config: ModelConfig | None = None,
    expected_manifest_hash: str | None = None,
    weights_only: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not weights_only:
        if expected_model_config and checkpoint["model_config"] != asdict(expected_model_config):
            raise ValueError("checkpoint model architecture does not match")
        if expected_manifest_hash and checkpoint["manifest_hash"] != expected_manifest_hash:
            raise ValueError("checkpoint manifest hash does not match")
    model.load_state_dict(checkpoint["model"])
    if optimizer and checkpoint["optimizer"] is not None and not weights_only:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

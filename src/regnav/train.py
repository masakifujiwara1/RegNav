import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader

from regnav.config import load_yaml_config
from regnav.data.collate import collate_regnav
from regnav.data.manifest import group_split, load_manifest
from regnav.data.vint_dataset import VintTrajectoryDataset
from regnav.models.factory import build_model
from regnav.training.engine import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    set_training_stage,
    train_epoch,
)
from regnav.training.sampler import DomainBalancedSampler


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    return torch.compile(model, mode="reduce-overhead") if enabled else model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RegNav model")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    model_config, training_config = load_yaml_config(
        args.model_config, args.training_config
    )
    seed_everything(training_config.seed)
    manifest_hash = sha256(args.manifest.read_bytes()).hexdigest()
    records = load_manifest(args.manifest)
    split = group_split(records, training_config.seed, (0.8, 0.1, 0.1))
    records = [record for record in records if split[record.trajectory_id] == args.split]
    if not records:
        raise ValueError(f"manifest has no {args.split} trajectories")

    dataset = VintTrajectoryDataset(records, model_config)
    labels = [records[record_index].dataset for record_index, _ in dataset.index]
    sampler = DomainBalancedSampler(labels, len(dataset), training_config.seed)
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        sampler=sampler,
        num_workers=training_config.workers,
        collate_fn=collate_regnav,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device)
    lora = [parameter for name, parameter in model.named_parameters() if "lora_" in name]
    regular = [parameter for name, parameter in model.named_parameters() if "lora_" not in name and parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": regular, "lr": training_config.learning_rate},
            {"params": lora, "lr": training_config.lora_learning_rate},
        ],
        weight_decay=training_config.weight_decay,
    )
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            expected_model_config=model_config,
            expected_manifest_hash=manifest_hash,
            weights_only=args.weights_only,
            map_location=device,
        )
        if not args.weights_only:
            start_epoch = checkpoint["epoch"]

    training_model = _maybe_compile(model, args.compile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    best = float("inf")
    for epoch in range(start_epoch, training_config.epochs):
        set_training_stage(model, epoch, training_config.lora_start_epoch)
        losses = train_epoch(training_model, loader, optimizer, training_config, device)
        with log_path.open("a") as file:
            file.write(json.dumps({"epoch": epoch + 1, **losses}) + "\n")
        checkpoint_args = dict(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            model_config=model_config,
            training_config=training_config,
            manifest_hash=manifest_hash,
            git_revision=_git_revision(),
        )
        save_checkpoint(args.output_dir / "last.pt", **checkpoint_args)
        if losses["total"] < best:
            best = losses["total"]
            save_checkpoint(args.output_dir / "best.pt", **checkpoint_args)


if __name__ == "__main__":
    main()

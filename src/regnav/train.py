import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader

from regnav.config import load_yaml_config
from regnav.data.collate import collate_regnav
from regnav.data.manifest import group_split, load_manifest, record_domain
from regnav.data.vint_dataset import VintTrajectoryDataset
from regnav.models.factory import build_model
from regnav.training.engine import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    set_training_stage,
    train_epoch,
    validation_ade,
)
from regnav.training.sampler import DomainBalancedSampler
from regnav.training.tracking import TrainingTracker, build_dataset_summary


_TRAINING_ARTIFACTS = ("metrics.jsonl", "best.pt", "last.pt")


def _ensure_output_dir_ready(output_dir: Path, resume: Path | None) -> None:
    if resume is not None:
        return
    conflicts = [output_dir / name for name in _TRAINING_ARTIFACTS if (output_dir / name).exists()]
    if conflicts:
        paths = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"output directory already contains training artifacts: {paths}; "
            "use --resume or choose a new directory"
        )


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    return torch.compile(model, mode="reduce-overhead") if enabled else model


def _summarize_validation_ade(values: dict[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("validation domains must not be empty")
    metrics = {
        f"validation/{domain.replace('/', '_')}_ade": value
        for domain, value in sorted(values.items())
    }
    metrics["validation/macro_ade"] = sum(values.values()) / len(values)
    metrics["validation/worst_ade"] = max(values.values())
    return metrics


def _checkpoint_metric(metrics: dict[str, float]) -> tuple[str, float]:
    if "validation/worst_ade" in metrics:
        return "validation/worst_ade", metrics["validation/worst_ade"]
    return "train/total", metrics["total"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a RegNav model")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-experiment", default="regnav")
    parser.add_argument("--mlflow-run-name")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _ensure_output_dir_ready(args.output_dir, args.resume)

    model_config, training_config = load_yaml_config(
        args.model_config, args.training_config
    )
    seed_everything(training_config.seed)
    manifest_hash = sha256(args.manifest.read_bytes()).hexdigest()
    all_records = load_manifest(args.manifest)
    split = group_split(all_records, training_config.seed, (0.8, 0.1, 0.1))
    records = [
        record for record in all_records if split[record.trajectory_id] == args.split
    ]
    if not records:
        raise ValueError(f"manifest has no {args.split} trajectories")

    dataset = VintTrajectoryDataset(records, model_config)
    dataset_summary = build_dataset_summary(
        family="vint",
        split=args.split,
        manifest_hash=manifest_hash,
        trajectory_subsets=[record_domain(record) for record in records],
        sample_subsets=[
            record_domain(records[record_index]) for record_index, _ in dataset.index
        ],
    )
    labels = [record_domain(records[record_index]) for record_index, _ in dataset.index]
    sampler = DomainBalancedSampler(labels, len(dataset), training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        sampler=sampler,
        num_workers=training_config.workers,
        collate_fn=collate_regnav,
        pin_memory=device.type == "cuda",
        persistent_workers=training_config.workers > 0,
    )
    validation_loaders: dict[str, DataLoader] = {}
    if args.split == "train":
        validation_records = defaultdict(list)
        for record in all_records:
            if split[record.trajectory_id] == "validation":
                validation_records[record_domain(record)].append(record)
        for domain, domain_records in sorted(validation_records.items()):
            validation_dataset = VintTrajectoryDataset(domain_records, model_config)
            if not len(validation_dataset):
                continue
            validation_loaders[domain] = DataLoader(
                validation_dataset,
                batch_size=training_config.batch_size,
                shuffle=False,
                num_workers=training_config.workers,
                collate_fn=collate_regnav,
                pin_memory=device.type == "cuda",
                persistent_workers=training_config.workers > 0,
            )

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
    best = float("inf")
    best_metric_name = (
        "validation/worst_ade" if validation_loaders else "train/total"
    )
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
            previous_best = checkpoint.get("best_metric")
            if (
                checkpoint.get("best_metric_name") == best_metric_name
                and previous_best is not None
            ):
                best = float(previous_best)

    training_model = _maybe_compile(model, args.compile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    with TrainingTracker(
        enabled=not args.no_mlflow,
        output_dir=args.output_dir,
        model_config=asdict(model_config),
        training_config=asdict(training_config),
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        split=args.split,
        manifest_hash=manifest_hash,
        device=str(device),
        compile_enabled=args.compile,
        git_revision=_git_revision(),
        tracking_uri=args.mlflow_tracking_uri,
        experiment=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        dataset_summary=dataset_summary,
    ) as tracker:
        for epoch in range(start_epoch, training_config.epochs):
            set_training_stage(model, epoch, training_config.lora_start_epoch)
            losses = train_epoch(training_model, loader, optimizer, training_config, device)
            metrics = dict(losses)
            if validation_loaders:
                metrics.update(
                    _summarize_validation_ade(
                        {
                            domain: validation_ade(
                                training_model, domain_loader, device
                            )
                            for domain, domain_loader in validation_loaders.items()
                        }
                    )
                )
            best_metric_name, checkpoint_metric = _checkpoint_metric(metrics)
            with log_path.open("a") as file:
                file.write(json.dumps({"epoch": epoch + 1, **metrics}) + "\n")
            tracker.log_epoch(epoch + 1, metrics)
            improved = checkpoint_metric < best
            if improved:
                best = checkpoint_metric
            checkpoint_args = dict(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                model_config=model_config,
                training_config=training_config,
                manifest_hash=manifest_hash,
                git_revision=_git_revision(),
                best_metric_name=best_metric_name,
                best_metric=best,
            )
            save_checkpoint(args.output_dir / "last.pt", **checkpoint_args)
            if improved:
                save_checkpoint(args.output_dir / "best.pt", **checkpoint_args)
        tracker.log_artifacts(
            final_epoch=training_config.epochs,
            best_metric=best,
            best_metric_name=best_metric_name,
        )


if __name__ == "__main__":
    main()

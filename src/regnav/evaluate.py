import argparse
from collections import defaultdict
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path

import torch

from regnav.baselines import constant_velocity, route_only
from regnav.config import ModelConfig
from regnav.data.collate import collate_regnav
from regnav.data.manifest import group_split, load_manifest
from regnav.data.vint_dataset import VintTrajectoryDataset
from regnav.metrics import trajectory_metrics
from regnav.models.factory import build_model
from regnav.training.engine import load_checkpoint
from regnav.training.privileged import collision_free_labels


def summarize_batch(
    prediction: torch.Tensor, batch: dict[str, torch.Tensor], interval: float
) -> dict[str, object]:
    target, mask = batch["target_trajectory"], batch["valid_mask"]
    baselines = {
        "constant_velocity": constant_velocity(batch["ego"], target.shape[1], interval),
        "route_only": route_only(batch["route_goal"], target.shape[1]),
    }
    model_metrics = trajectory_metrics(prediction, target, mask, interval)
    result: dict[str, object] = {"model": model_metrics}
    for name, trajectory in baselines.items():
        result[name] = trajectory_metrics(trajectory, target, mask, interval)
    result["meets_baseline_acceptance"] = (
        model_metrics["ade"] < result["constant_velocity"]["ade"]  # type: ignore[index]
        and model_metrics["fde"] < result["constant_velocity"]["fde"]  # type: ignore[index]
        and model_metrics["ade"] < result["route_only"]["ade"]  # type: ignore[index]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RegNav checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    records = load_manifest(args.manifest)
    split = group_split(records, checkpoint["training_config"]["seed"], (0.8, 0.1, 0.1))
    records = [record for record in records if split[record.trajectory_id] == args.split]
    dataset = VintTrajectoryDataset(records, model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device).eval()
    load_checkpoint(args.checkpoint, model, weights_only=True, map_location=device)

    predictions = defaultdict(list)
    targets = defaultdict(list)
    masks = defaultdict(list)
    egos = defaultdict(list)
    routes = defaultdict(list)
    selected_collisions = []
    proposal_collisions = []
    with torch.no_grad():
        for index in range(len(dataset)):
            record_index, _ = dataset.index[index]
            name = records[record_index].dataset
            batch = {key: value.to(device) for key, value in collate_regnav([dataset[index]]).items()}
            output = model(batch)
            predictions[name].append(output["trajectory"].cpu())
            targets[name].append(batch["target_trajectory"].cpu())
            masks[name].append(batch["valid_mask"].cpu())
            egos[name].append(batch["ego"].cpu())
            routes[name].append(batch["route_goal"].cpu())
            if "obstacle_points" in batch:
                free = collision_free_labels(
                    output["proposals"],
                    batch["obstacle_points"],
                    batch["obstacle_points_mask"],
                    footprint=(1.0, 0.7),
                )
                selected_collisions.append((1 - free.gather(1, output["scores"].argmax(-1, keepdim=True))).cpu())
                proposal_collisions.append((1 - free).cpu())

    reports = {}
    for name in predictions:
        batch = {
            "target_trajectory": torch.cat(targets[name]),
            "valid_mask": torch.cat(masks[name]),
            "ego": torch.cat(egos[name]),
            "route_goal": torch.cat(routes[name]),
        }
        reports[name] = summarize_batch(
            torch.cat(predictions[name]), batch, model_config.interval
        )
    all_prediction = torch.cat([item for values in predictions.values() for item in values])
    aggregate_batch = {
        "target_trajectory": torch.cat([item for values in targets.values() for item in values]),
        "valid_mask": torch.cat([item for values in masks.values() for item in values]),
        "ego": torch.cat([item for values in egos.values() for item in values]),
        "route_goal": torch.cat([item for values in routes.values() for item in values]),
    }
    aggregate = summarize_batch(all_prediction, aggregate_batch, model_config.interval)
    aggregate["selected_collision_rate"] = (
        float(torch.cat(selected_collisions).mean()) if selected_collisions else None
    )
    aggregate["mean_proposal_collision_rate"] = (
        float(torch.cat(proposal_collisions).mean()) if proposal_collisions else None
    )
    report = {
        "model_config": checkpoint["model_config"],
        "checkpoint_hash": sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": args.split,
        "aggregate": aggregate,
        "datasets": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

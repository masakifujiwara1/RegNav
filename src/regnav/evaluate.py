import argparse
from collections import defaultdict
from dataclasses import fields
from hashlib import sha256
import heapq
import json
from pathlib import Path
import re

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
from regnav.visualization import render_trajectory_png, write_visualization_index


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a RegNav checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualize-dir", type=Path)
    parser.add_argument("--visualize-count", type=_non_negative_int, default=12)
    parser.add_argument("--visualize-worst-k", type=_non_negative_int, default=0)
    return parser


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "sample"


def _visualization_sample(
    record: object,
    frame: int,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    interval: float,
    selected_collision: float | None,
) -> dict[str, object]:
    prediction = output["trajectory"].detach().cpu()
    sample_batch = {key: value.detach().cpu() for key, value in batch.items()}
    target = sample_batch["target_trajectory"]
    metrics = summarize_batch(prediction, sample_batch, interval)
    sample: dict[str, object] = {
        "dataset": record.dataset,
        "trajectory_id": record.trajectory_id,
        "frame": frame,
        "image": sample_batch["image"][0],
        "target": target[0],
        "prediction": prediction[0],
        "constant_velocity": constant_velocity(
            sample_batch["ego"], target.shape[1], interval
        )[0],
        "route_only": route_only(sample_batch["route_goal"], target.shape[1])[0],
        "metrics": metrics,
    }
    if "obstacle_points" in sample_batch:
        mask = sample_batch["obstacle_points_mask"][0]
        sample["obstacles"] = sample_batch["obstacle_points"][0][mask]
    if selected_collision is not None:
        sample["selected_collision"] = selected_collision
    return sample


def _write_visualizations(output_dir: Path, samples: list[dict[str, object]]) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = []
    for index, sample in enumerate(samples, 1):
        name = (
            f"{index:04d}-{_safe_name(sample['dataset'])}-"
            f"{_safe_name(sample['trajectory_id'])}-frame-{sample['frame']}.png"
        )
        render_trajectory_png(sample, output_dir / name)
        image_files.append(name)
    return write_visualization_index(output_dir, samples, image_files)


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
    args = build_parser().parse_args()

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
    visualization_samples: list[dict[str, object]] = []
    worst_visualizations: list[tuple[float, int, dict[str, object]]] = []
    visualization_counter = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            record_index, frame = dataset.index[index]
            record = records[record_index]
            name = record.dataset
            batch = {key: value.to(device) for key, value in collate_regnav([dataset[index]]).items()}
            output = model(batch)
            predictions[name].append(output["trajectory"].cpu())
            targets[name].append(batch["target_trajectory"].cpu())
            masks[name].append(batch["valid_mask"].cpu())
            egos[name].append(batch["ego"].cpu())
            routes[name].append(batch["route_goal"].cpu())
            selected_collision = None
            if "obstacle_points" in batch:
                free = collision_free_labels(
                    output["proposals"],
                    batch["obstacle_points"],
                    batch["obstacle_points_mask"],
                    footprint=(1.0, 0.7),
                )
                selected = 1 - free.gather(1, output["scores"].argmax(-1, keepdim=True))
                selected_collision = float(selected[0, 0])
                selected_collisions.append(selected.cpu())
                proposal_collisions.append((1 - free).cpu())
            if args.visualize_dir is not None and (
                index < args.visualize_count or args.visualize_worst_k
            ):
                sample = _visualization_sample(
                    record,
                    frame,
                    output,
                    batch,
                    model_config.interval,
                    selected_collision,
                )
                if index < args.visualize_count:
                    visualization_samples.append(sample)
                if args.visualize_worst_k:
                    model_metrics = sample["metrics"]["model"]
                    ade = float(model_metrics["ade"])
                    entry = (ade, visualization_counter, sample)
                    if len(worst_visualizations) < args.visualize_worst_k:
                        heapq.heappush(worst_visualizations, entry)
                    elif entry[0] > worst_visualizations[0][0]:
                        heapq.heapreplace(worst_visualizations, entry)
                    visualization_counter += 1

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
    if args.visualize_dir is not None:
        selected_samples = list(visualization_samples)
        seen = {(sample["trajectory_id"], sample["frame"]) for sample in selected_samples}
        for _, _, sample in sorted(
            worst_visualizations, key=lambda item: (-item[0], item[1])
        ):
            key = (sample["trajectory_id"], sample["frame"])
            if key not in seen:
                selected_samples.append(sample)
                seen.add(key)
        _write_visualizations(args.visualize_dir, selected_samples)
        report["visualizations"] = {
            "directory": str(args.visualize_dir),
            "index": "index.html",
            "samples": "samples.jsonl",
            "count": len(selected_samples),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

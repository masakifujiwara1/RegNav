import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import torch

from regnav.config import ModelConfig
from regnav.data.collate import collate_regnav
from regnav.data.manifest import group_split, load_manifest
from regnav.data.vint_dataset import VintTrajectoryDataset
from regnav.evaluate import _evaluation_index
from regnav.models.factory import build_model
from regnav.streaming import evaluate_cache_intervals
from regnav.training.engine import load_checkpoint


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate cached scene intervals")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-refresh-intervals",
        type=_positive_int,
        nargs="+",
        default=[1, 2, 3, 5, 10],
    )
    parser.add_argument("--minimum-frame", type=_positive_int)
    parser.add_argument("--device")
    parser.add_argument(
        "--turn-rate-threshold", type=_non_negative_float, default=0.3
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    records = load_manifest(args.manifest)
    split = group_split(records, checkpoint["training_config"]["seed"], (0.8, 0.1, 0.1))
    records = [record for record in records if split[record.trajectory_id] == args.split]
    dataset = VintTrajectoryDataset(records, config)
    dataset.index = _evaluation_index(dataset.index, args.minimum_frame)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(config).to(device).eval()
    if not hasattr(model, "encode_scene") or not hasattr(model, "decode_scene"):
        raise ValueError("checkpoint model does not support cached scene inference")
    load_checkpoint(args.checkpoint, model, weights_only=True, map_location=device)

    def samples():
        for index, (record_index, _) in enumerate(dataset.index):
            batch = {
                key: value.to(device)
                for key, value in collate_regnav([dataset[index]]).items()
            }
            yield records[record_index].trajectory_id, batch

    intervals = tuple(dict.fromkeys((1, *args.scene_refresh_intervals)))
    results = evaluate_cache_intervals(
        model,
        samples(),
        scene_refresh_intervals=intervals,
        trajectory_interval=config.interval,
        device=device,
        turn_rate_threshold=args.turn_rate_threshold,
    )
    report = {
        "model_config": checkpoint["model_config"],
        "checkpoint_hash": sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": args.split,
        "minimum_frame": args.minimum_frame,
        "samples": len(dataset),
        "device": str(device),
        "scene_refresh_intervals": list(intervals),
        "turn_rate_threshold": args.turn_rate_threshold,
        "intervals": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

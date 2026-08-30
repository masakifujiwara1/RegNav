import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from regnav.data.manifest import group_split, load_manifest
from regnav.config import ModelConfig
from regnav.data.vint_dataset import VintTrajectoryDataset, preprocess_image
from regnav.models.factory import build_model
from regnav.streaming import CachedSceneInference
from regnav.training.engine import load_checkpoint


def _image_shape(config: ModelConfig) -> tuple[int, ...]:
    width, height = config.image_size
    if config.context_frames > 1:
        return (1, 3, config.context_frames, height, width)
    return (1, 3, height, width)


def _percentiles(durations: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "p99_ms": float(np.percentile(durations, 99)),
    }


def _measure_operations(
    operations: dict,
    device: torch.device,
    iterations: int,
    warmup: int,
) -> dict[str, dict[str, float]]:
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        for operation in operations.values():
            operation()
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    items = list(operations.items())
    durations = {name: [] for name in operations}
    for iteration in range(iterations):
        offset = iteration % len(items)
        for name, operation in items[offset:] + items[:offset]:
            synchronize()
            start = perf_counter()
            operation()
            synchronize()
            durations[name].append((perf_counter() - start) * 1000)
    return {name: _percentiles(values) for name, values in durations.items()}


def _benchmark_stages(
    model,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    iterations: int,
    warmup: int,
    scene_refresh_interval: int | None = None,
) -> dict[str, dict[str, float]]:
    operations = {"end_to_end": lambda: model(batch)}
    with torch.inference_mode():
        if hasattr(model, "encode_scene") and hasattr(model, "decode_scene"):
            scene = model.encode_scene(batch["image"])
            operations.update(
                {
                    "scene_encoder": lambda: model.encode_scene(batch["image"]),
                    "cached_head": lambda: model.decode_scene(
                        scene, batch["ego"], batch["route_goal"]
                    ),
                }
            )
            if scene_refresh_interval is not None:
                runner = CachedSceneInference(model, scene_refresh_interval)
                operations["streaming"] = lambda: runner(batch)

        return _measure_operations(
            operations, device, iterations=iterations, warmup=warmup
        )


class _TimedModel:
    def __init__(self, model):
        self.model = model
        self.encoder_ms = 0.0
        self.head_ms = 0.0
        self.encoder_calls = 0

    def eval(self):
        self.model.eval()
        return self

    def _measure(self, operation):
        start = perf_counter()
        output = operation()
        return output, (perf_counter() - start) * 1000

    def encode_scene(self, image):
        output, self.encoder_ms = self._measure(
            lambda: self.model.encode_scene(image)
        )
        self.encoder_calls += 1
        return output

    def decode_scene(self, scene, ego, route_goal):
        output, self.head_ms = self._measure(
            lambda: self.model.decode_scene(scene, ego, route_goal)
        )
        return output


def _preprocess_paths(
    image_paths: tuple[Path, ...],
    image_size: tuple[int, int],
) -> torch.Tensor:
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(preprocess_image(image, image_size))
    return (
        images[0]
        if len(images) == 1
        else torch.stack(images, dim=1)
    )


def _benchmark_replay(
    model,
    frames: Sequence[tuple[tuple[Path, ...], torch.Tensor, torch.Tensor]],
    *,
    image_size: tuple[int, int],
    device: torch.device,
    scene_refresh_interval: int,
    turn_rate_threshold: float,
) -> dict[str, object]:
    if device.type != "cpu":
        raise ValueError("real image replay benchmark supports CPU only")
    if not frames:
        raise ValueError("frames must not be empty")
    model.eval()
    warmup_paths, warmup_ego, warmup_route_goal = frames[0]
    warmup_batch = {
        "image": _preprocess_paths(warmup_paths, image_size)[None].to(device),
        "ego": warmup_ego[None].to(device),
        "route_goal": warmup_route_goal[None].to(device),
    }
    with torch.inference_mode():
        warmup_scene = model.encode_scene(warmup_batch["image"])
        model.decode_scene(
            warmup_scene, warmup_batch["ego"], warmup_batch["route_goal"]
        )
    timed_model = _TimedModel(model)
    runner = CachedSceneInference(
        timed_model,
        scene_refresh_interval,
        turn_rate_threshold,
    )
    durations = {
        name: []
        for name in (
            "image_preprocess",
            "input_transfer",
            "scene_encoder",
            "cached_head",
            "streaming_model",
            "end_to_end",
        )
    }

    for image_paths, ego, route_goal in frames:
        end_to_end_start = perf_counter()

        start = perf_counter()
        image_tensor = _preprocess_paths(image_paths, image_size)
        durations["image_preprocess"].append((perf_counter() - start) * 1000)

        start = perf_counter()
        batch = {
            "image": image_tensor[None].to(device),
            "ego": ego[None].to(device),
            "route_goal": route_goal[None].to(device),
        }
        durations["input_transfer"].append((perf_counter() - start) * 1000)

        encoder_calls = timed_model.encoder_calls
        start = perf_counter()
        runner(batch)
        model_ms = (perf_counter() - start) * 1000
        if timed_model.encoder_calls > encoder_calls:
            durations["scene_encoder"].append(timed_model.encoder_ms)
        durations["cached_head"].append(timed_model.head_ms)
        durations["streaming_model"].append(model_ms)
        durations["end_to_end"].append((perf_counter() - end_to_end_start) * 1000)

    stages = {
        name: {"measurements": len(values), **_percentiles(values)}
        for name, values in durations.items()
    }
    return {
        "samples": len(frames),
        "scene_refreshes": timed_model.encoder_calls,
        "refresh_rate": timed_model.encoder_calls / len(frames),
        "scene_refresh_interval": scene_refresh_interval,
        "turn_rate_threshold": turn_rate_threshold,
        "warmup_frames": 1,
        "image_io": "warm_os_cache",
        "stages": stages,
    }


def _replay_frames(dataset, records, samples: int):
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not dataset.index:
        raise ValueError("replay split contains no frames")
    first_record_index = dataset.index[0][0]
    selected = [
        (dataset_index, frame)
        for dataset_index, (record_index, frame) in enumerate(dataset.index)
        if record_index == first_record_index
    ][:samples]
    record = records[first_record_index]
    root = Path(record.path)
    frames = []
    for dataset_index, frame in selected:
        sample = dataset[dataset_index]
        image_paths = tuple(
            root / f"{image_index}.jpg"
            for image_index in range(
                frame - dataset.config.context_frames + 1,
                frame + 1,
            )
        )
        frames.append((image_paths, sample["ego"], sample["route_goal"]))
    return frames, record.trajectory_id


def _replay_samples(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark RegNav batch-1 latency")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--scene-refresh-interval", type=int, default=5)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--samples", type=_replay_samples, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--turn-rate-threshold", type=float, default=0.3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    device = torch.device(args.device)
    model = build_model(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, weights_only=True, map_location=device)
    if args.manifest is not None:
        records = load_manifest(args.manifest)
        split = group_split(
            records,
            checkpoint["training_config"]["seed"],
            (0.8, 0.1, 0.1),
        )
        records = [
            record
            for record in records
            if split[record.trajectory_id] == args.split
        ]
        dataset = VintTrajectoryDataset(records, config)
        frames, trajectory_id = _replay_frames(
            dataset, records, args.samples
        )
        result = _benchmark_replay(
            model,
            frames,
            image_size=config.image_size,
            device=device,
            scene_refresh_interval=args.scene_refresh_interval,
            turn_rate_threshold=args.turn_rate_threshold,
        )
        result = {
            "mode": "real_image_replay",
            "device": str(device),
            "split": args.split,
            "trajectory_id": trajectory_id,
            **result,
        }
        serialized = json.dumps(result, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        print(serialized)
        return

    batch = {
        "image": torch.randn(_image_shape(config), device=device),
        "ego": torch.zeros(1, config.ego_features, device=device),
        "route_goal": torch.zeros(1, config.route_features, device=device),
        "target_trajectory": torch.zeros(1, config.num_poses, 3, device=device),
        "valid_mask": torch.ones(1, config.num_poses, dtype=torch.bool, device=device),
    }

    stages = _benchmark_stages(
        model, batch, device, args.iterations, args.warmup,
        scene_refresh_interval=args.scene_refresh_interval,
    )

    result = {
        "device": str(device),
        "iterations": args.iterations,
        **stages["end_to_end"],
        "stages": stages,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

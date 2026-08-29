import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from regnav.config import ModelConfig
from regnav.models.factory import build_model
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
        return _measure_operations(
            operations, device, iterations=iterations, warmup=warmup
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RegNav batch-1 latency")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    device = torch.device(args.device)
    model = build_model(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, weights_only=True, map_location=device)
    batch = {
        "image": torch.randn(_image_shape(config), device=device),
        "ego": torch.zeros(1, config.ego_features, device=device),
        "route_goal": torch.zeros(1, config.route_features, device=device),
        "target_trajectory": torch.zeros(1, config.num_poses, 3, device=device),
        "valid_mask": torch.ones(1, config.num_poses, dtype=torch.bool, device=device),
    }

    stages = _benchmark_stages(model, batch, device, args.iterations, args.warmup)

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

from collections.abc import Iterable
from itertools import chain
from time import perf_counter

import numpy as np
import torch

from regnav.contracts import RegNavBatch, RegNavOutput
from regnav.metrics import ade, fde, heading_error, trajectory_metrics


class CachedSceneInference:
    def __init__(self, model, scene_refresh_interval: int):
        if scene_refresh_interval <= 0:
            raise ValueError("scene_refresh_interval must be positive")
        self.model = model.eval()
        self.scene_refresh_interval = scene_refresh_interval
        self.scene = None
        self.frame_index = 0

    @torch.inference_mode()
    def __call__(self, batch: RegNavBatch) -> RegNavOutput:
        if self.scene is None or self.frame_index % self.scene_refresh_interval == 0:
            self.scene = self.model.encode_scene(batch["image"])
        output = self.model.decode_scene(
            self.scene, batch["ego"], batch["route_goal"]
        )
        self.frame_index += 1
        return output


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _trajectory_delta(
    prediction: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    return {
        "ade": float(ade(prediction, baseline, mask)),
        "fde": float(fde(prediction, baseline, mask)),
        "heading_error": float(heading_error(prediction, baseline, mask)),
    }


def evaluate_cache_intervals(
    model,
    samples: Iterable[tuple[str, RegNavBatch]],
    scene_refresh_intervals: tuple[int, ...],
    trajectory_interval: float,
    device: torch.device,
    turn_rate_threshold: float = 0.3,
) -> dict[int, dict[str, object]]:
    model.eval()
    intervals = tuple(dict.fromkeys((1, *scene_refresh_intervals)))
    caches = {interval: None for interval in intervals}
    predictions: dict[int, list[torch.Tensor]] = {interval: [] for interval in intervals}
    latencies: dict[int, list[float]] = {interval: [] for interval in intervals}
    refreshes = {interval: 0 for interval in intervals}
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    turning_samples: list[torch.Tensor] = []
    previous_trajectory = None
    frame = 0
    measurement_frame = 0
    iterator = iter(samples)
    try:
        first_sample = next(iterator)
    except StopIteration as error:
        raise ValueError("samples must not be empty") from error

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        _, warmup_batch = first_sample
        scene = model.encode_scene(warmup_batch["image"])
        for _ in intervals:
            model.decode_scene(
                scene, warmup_batch["ego"], warmup_batch["route_goal"]
            )
        synchronize()

        for trajectory_id, batch in chain((first_sample,), iterator):
            if trajectory_id != previous_trajectory:
                caches = {interval: None for interval in intervals}
                frame = 0
                previous_trajectory = trajectory_id

            synchronize()
            start = perf_counter()
            scene = model.encode_scene(batch["image"])
            synchronize()
            encoder_ms = (perf_counter() - start) * 1000

            offset = measurement_frame % len(intervals)
            ordered_intervals = intervals[offset:] + intervals[:offset]
            for interval in ordered_intervals:
                refresh = frame % interval == 0
                if refresh:
                    caches[interval] = scene
                    refreshes[interval] += 1
                synchronize()
                start = perf_counter()
                output = model.decode_scene(
                    caches[interval], batch["ego"], batch["route_goal"]
                )
                synchronize()
                head_ms = (perf_counter() - start) * 1000
                latencies[interval].append(head_ms + (encoder_ms if refresh else 0))
                predictions[interval].append(output["trajectory"].cpu())

            targets.append(batch["target_trajectory"].cpu())
            masks.append(batch["valid_mask"].cpu())
            turning_samples.append(
                (batch["ego"][:, 1].abs() >= turn_rate_threshold).cpu()
            )
            frame += 1
            measurement_frame += 1

    target = torch.cat(targets)
    mask = torch.cat(masks)
    turning = torch.cat(turning_samples)
    full = torch.cat(predictions[1])
    report = {}
    for interval in intervals:
        prediction = torch.cat(predictions[interval])
        turning_report = None
        if turning.any():
            turning_report = {
                "samples": int(turning.sum()),
                "target": trajectory_metrics(
                    prediction[turning], target[turning], mask[turning], trajectory_interval
                ),
                "delta_to_interval_1": _trajectory_delta(
                    prediction[turning], full[turning], mask[turning]
                ),
            }
        report[interval] = {
            "target": trajectory_metrics(
                prediction, target, mask, trajectory_interval
            ),
            "delta_to_interval_1": _trajectory_delta(prediction, full, mask),
            "turning": turning_report,
            "refresh_rate": refreshes[interval] / len(predictions[interval]),
            "latency_ms": _latency_summary(latencies[interval]),
        }
    return report

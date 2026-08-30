from collections.abc import Iterable
from itertools import chain
from math import isfinite
from time import perf_counter

import numpy as np
import torch

from regnav.contracts import RegNavBatch, RegNavOutput
from regnav.metrics import ade, fde, heading_error, trajectory_metrics


class CachedSceneInference:
    def __init__(
        self,
        model,
        scene_refresh_interval: int,
        turn_rate_threshold: float | None = None,
    ):
        if scene_refresh_interval <= 0:
            raise ValueError("scene_refresh_interval must be positive")
        if turn_rate_threshold is not None and (
            not isfinite(turn_rate_threshold) or turn_rate_threshold < 0
        ):
            raise ValueError("turn_rate_threshold must be non-negative")
        self.model = model.eval()
        self.scene_refresh_interval = scene_refresh_interval
        self.turn_rate_threshold = turn_rate_threshold
        self.scene = None
        self.frame_index = 0

        self.last_scene_refresh = None
    @torch.inference_mode()
    def __call__(self, batch: RegNavBatch) -> RegNavOutput:
        turning = self.turn_rate_threshold is not None and bool(
            (batch["ego"][:, 1].abs() >= self.turn_rate_threshold).any()
        )
        if (
            self.last_scene_refresh is None
            or self.frame_index - self.last_scene_refresh
            >= self.scene_refresh_interval
            or turning
        ):
            self.scene = self.model.encode_scene(batch["image"])
            self.last_scene_refresh = self.frame_index
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
    adaptive_scene_refresh_interval: int | None = None,
) -> dict[int | str, dict[str, object]]:
    if (
        adaptive_scene_refresh_interval is not None
        and adaptive_scene_refresh_interval <= 0
    ):
        raise ValueError("adaptive_scene_refresh_interval must be positive")
    if not isfinite(turn_rate_threshold) or turn_rate_threshold < 0:
        raise ValueError("turn_rate_threshold must be finite and non-negative")
    model.eval()
    intervals = tuple(dict.fromkeys((1, *scene_refresh_intervals)))
    policy_intervals: dict[int | str, int] = {interval: interval for interval in intervals}
    if adaptive_scene_refresh_interval is not None:
        policy_intervals[f"adaptive_{adaptive_scene_refresh_interval}"] = (
            adaptive_scene_refresh_interval
        )
    policies = tuple(policy_intervals)
    caches = {policy: None for policy in policies}
    last_refreshes = {policy: None for policy in policies}
    predictions = {policy: [] for policy in policies}
    latencies = {policy: [] for policy in policies}
    refreshes = {policy: 0 for policy in policies}
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
        for _ in policies:
            model.decode_scene(
                scene, warmup_batch["ego"], warmup_batch["route_goal"]
            )
        synchronize()

        for trajectory_id, batch in chain((first_sample,), iterator):
            if trajectory_id != previous_trajectory:
                caches = {policy: None for policy in policies}
                last_refreshes = {policy: None for policy in policies}
                frame = 0
                previous_trajectory = trajectory_id

            synchronize()
            start = perf_counter()
            scene = model.encode_scene(batch["image"])
            synchronize()
            encoder_ms = (perf_counter() - start) * 1000

            turning = bool(
                (batch["ego"][:, 1].abs() >= turn_rate_threshold).any()
            )
            offset = measurement_frame % len(policies)
            ordered_policies = policies[offset:] + policies[:offset]
            for policy in ordered_policies:
                interval = policy_intervals[policy]
                last_refresh = last_refreshes[policy]
                refresh = (
                    last_refresh is None
                    or frame - last_refresh >= interval
                    or isinstance(policy, str) and turning
                )
                if refresh:
                    caches[policy] = scene
                    last_refreshes[policy] = frame
                    refreshes[policy] += 1
                synchronize()
                start = perf_counter()
                output = model.decode_scene(
                    caches[policy], batch["ego"], batch["route_goal"]
                )
                synchronize()
                head_ms = (perf_counter() - start) * 1000
                latencies[policy].append(head_ms + (encoder_ms if refresh else 0))
                predictions[policy].append(output["trajectory"].cpu())

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
    for policy in policies:
        prediction = torch.cat(predictions[policy])
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
        report[policy] = {
            "target": trajectory_metrics(
                prediction, target, mask, trajectory_interval
            ),
            "delta_to_interval_1": _trajectory_delta(prediction, full, mask),
            "turning": turning_report,
            "refresh_rate": refreshes[policy] / len(predictions[policy]),
            "latency_ms": _latency_summary(latencies[policy]),
        }
    return report
